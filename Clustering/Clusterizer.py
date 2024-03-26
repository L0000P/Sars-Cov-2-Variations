import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import gc
import seaborn as sns
from pyclustering.cluster.cure import cure
from scipy.sparse import issparse
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering, DBSCAN, OPTICS, HDBSCAN, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.impute import SimpleImputer

class Clusterizer:
    def __init__(self, clt_key="km", sample_frac=0.5): # Constructor
        self.clusters_type = {"km": KMeans, "ka": KMedoids, 
                              "gc": AgglomerativeClustering, "db": DBSCAN, 
                              "op": OPTICS, "hdb": HDBSCAN,
                              "cc": cure} # Supported clustering types
        self.sample_frac = sample_frac # Fraction of samples to use
        if clt_key not in self.clusters_type: # Check if the clustering type is supported
            raise ValueError(f"Clustering type {clt_key} is not supported.")
        self.clt_key = clt_key # Clustering type

    def load_and_sample_in_chunks(self, file_path, chunksize=10000): # Load and sample data in chunks
        cols_to_use = ['Geo_Location', 'Mutations', 'Clade'] # Columns to use
        reader = pd.read_csv(file_path, usecols=cols_to_use, low_memory=False, chunksize=chunksize) # Read CSV file in chunks
        
        for df_chunk in reader: # Iterate over the chunks
            if 0 < self.sample_frac < 1.0: # If the sample fraction is between 0 and 1
                df_chunk = df_chunk.sample(frac=self.sample_frac) # Sample the data
            yield df_chunk # Yield the chunk

    def extract_all_mutations(self, mutations_list): # Extract all mutations
        all_mutations = set() # Set of all mutations
        for mutations in mutations_list: # Iterate over the mutations
            all_mutations.update(mutations) # Update the set of all mutations
        return all_mutations # Return the set of all mutations

    def sanitize_file_name(self, file_name): # Sanitize the file name
        parts = file_name.split('_') # Split the file name
        sanitized_parts = [] # List of sanitized parts
        seen = set() # Set of seen parts

        for part in parts: # Iterate over the parts
            clean_part = part.strip() # Clean the part
            if clean_part not in seen and clean_part != '': # If the part is not seen and not empty
                sanitized_parts.append(clean_part) # Append the clean part
                seen.add(clean_part) # Add the clean part to the set of seen parts

        sanitized_file_name = '_'.join(sanitized_parts) # Join the sanitized parts
        return sanitized_file_name # Return the sanitized file name

    def mutations_to_sparse_matrix(self, mutations_list, all_mutations): # Convert mutations to a sparse matrix
        mutation_index = {mutation: i for i, mutation in enumerate(sorted(all_mutations))} # Create a mutation index
        rows = [] # List of rows
        cols = [] # List of columns
        data = [] # List of data
        for i, mutations in enumerate(mutations_list): # Iterate over the mutations
            for mutation in mutations: # Iterate over the mutation
                if mutation in mutation_index: # If the mutation is in the mutation index
                    rows.append(i) # Append the row
                    cols.append(mutation_index[mutation]) # Append the column
                    data.append(1) # Append the data

        X_sparse = csr_matrix((data, (rows, cols)), shape=(len(mutations_list), len(mutation_index)), dtype=int) # Create a sparse matrix
        return X_sparse # Return the sparse matrix

    def calculate_jaccard_distance(self, X): # Calculate Jaccard distance
        X_csr = X.tocsr() # Convert X to CSR format

        intersect = X_csr.dot(X_csr.T) # Calculate the intersection
        sizes = X_csr.sum(axis=1).A1 # Calculate the sizes
        union = sizes[:, None] + sizes - intersect.toarray() # Calculate the union
        jaccard_similarity = intersect.toarray() / union # Calculate the Jaccard similarity
        jaccard_distance = 1 - jaccard_similarity # Calculate the Jaccard distance

        return jaccard_distance

    def process_file(self, input_folder, plot_folder, file_name): # Process the file
        print(f"Processing {file_name} using {self.clt_key.upper()}...") # Print the file name
        file_path = os.path.join(input_folder, file_name) # Get the file path
        for df_chunk in self.load_and_sample_in_chunks(file_path): # Iterate over the chunks
            if df_chunk.empty or len(df_chunk) < 2: continue # If the chunk is empty or has less than 2 rows, continue

            unique_geo_locations = df_chunk['Geo_Location'].unique() # Get the unique geo locations
            for geo_location in unique_geo_locations: # Iterate over the unique geo locations
                df_geo = df_chunk[df_chunk['Geo_Location'] == geo_location] # Get the geo location data
                if not df_geo.empty: # If the geo location data is not empty
                    geo_file_name = f"{geo_location.replace('/', '_').replace(':', '_')}_{file_name}" # Get the geo file name
                    self.cluster_and_plot(df_geo, plot_folder, geo_file_name) # Cluster and plot the geo location data

    def cluster_and_plot(self, df, plot_folder, file_name): # Cluster and plot the data
        sanitized_file_name = self.sanitize_file_name(file_name.replace('.csv', '')) + f'_{self.clt_key}_clusters.png' # Sanitize the file name
        
        mutations_list = df['Mutations'].apply(lambda x: set(x.split(','))).tolist() # Get the mutations list
        all_mutations = self.extract_all_mutations(mutations_list) # Extract all mutations
        X_sparse = self.mutations_to_sparse_matrix(mutations_list, all_mutations) # Convert mutations to a sparse matrix
        distance_matrix = self.calculate_jaccard_distance(X_sparse) # Calculate the Jaccard distance
        
        if self.clt_key == "gc": # If the clustering type is Agglomerative Clustering
            print(f"Clustering {sanitized_file_name} using Agglomerative Clustering...")
            clade_labels = df['Clade'].values.tolist() # Get the clade labels
            self.plot_dendrogram(distance_matrix, clade_labels, plot_folder, sanitized_file_name) # Plot the dendrogram
        else:            
            labels, num_clusters = self.apply_clustering(df, distance_matrix) # Apply clustering
            
            print(f"Clustering {sanitized_file_name}...") 
            if labels is not None:
                print(f"Plotting {sanitized_file_name}...") # Print plot message
                self.plot_results(distance_matrix, labels, plot_folder, sanitized_file_name, df) # Plot the results
            else:
                print(f"Skipping clustering for {sanitized_file_name} due to insufficient data.")   # Print skip message

    def apply_clustering(self, df, distance_matrix): # Apply clustering
        num_clusters, num_samples = self.prepare_clustering(df, distance_matrix) # Prepare clustering
        if num_samples < num_clusters: # If the number of samples is less than the number of clusters
            num_clusters = self.adjust_clusters(num_samples) # Adjust clusters
        return self.perform_clustering(distance_matrix, num_clusters) # Perform clustering

    def prepare_clustering(self, df, distance_matrix): # Prepare clustering
        num_clusters = len(set(df["Clade"])) # Get the number of clusters
        num_samples = distance_matrix.shape[0] # Get the number of samples
        return num_clusters, num_samples # Return the number of clusters and samples

    def adjust_clusters(self, num_samples): # Adjust clusters
        return max(2, num_samples // 2) # Return the maximum of 2 and the number of samples divided by 2

    def perform_clustering(self, distance_matrix, num_clusters): # Perform clustering
        svd_result = self.apply_svd(distance_matrix) # Apply SVD
        if svd_result is not None: # If the SVD result is not None
            
            if self.clt_key == "cc": # If the clustering type is CURE
                cure_instance = cure(svd_result.tolist(), number_cluster=num_clusters) # Get the CURE instance
                cure_instance.process() # Process the CURE instance
                clusters = cure_instance.get_clusters() # Get the clusters
                labels = [0] * len(svd_result) # Initialize the labels
                for cluster_idx, cluster in enumerate(clusters): # Iterate over the clusters
                    for index in cluster: # Iterate over the cluster indices
                        labels[index] = cluster_idx # Set the label
                return labels, num_clusters # Return the labels and the number of clusters
            else:
                num_samples = svd_result.shape[0] # Get the number of samples
                model = self.clusters_type[self.clt_key]() # Get the clustering model
                
                if self.clt_key == "op": # If the clustering type is OPTICS
                    min_samples = min(5, num_samples) # Get the minimum number of samples
                    model.set_params(min_samples=min_samples) # Set the minimum number of samples

                if self.clt_key == "hdb": # If the clustering type is HDBSCAN
                    min_samples = min(5, num_samples) # Get the minimum number of samples
                    min_cluster_size = min(5, num_samples)  # Get the minimum cluster size
                    model = self.clusters_type[self.clt_key](min_samples=min_samples, min_cluster_size=min_cluster_size) # Get the clustering model

                if self.clt_key in ["km", "ka"]: # If the clustering type is KMeans or KMedoids
                    model = self.clusters_type[self.clt_key](n_clusters=min(num_clusters, num_samples)) # Get the clustering model

                return model.fit_predict(svd_result), num_clusters # Return the model prediction and the number of clusters
        return None, num_clusters  # Return None and the number of clusters

    def apply_svd(self, distance_matrix): # Apply SVD
        try: # Try to apply SVD
            svd = TruncatedSVD(n_components=2, random_state=42) # Get the SVD model
            return svd.fit_transform(SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(distance_matrix)) # Return the SVD result
        except Exception as e: # Handle the exception
            print(f"Error in SVD: {e}") # Print the error
            return None # Return None

    def plot_results(self, distance_matrix, labels, plot_folder, file_name, df): # Plot the results
        if not self.check_min_points(labels): # Check if there are enough points to plot
            print(f"Not enough points to plot for {file_name}.") # Print the message
            return # Return
        pca_result = self.apply_pca(distance_matrix) # Apply PCA
        if pca_result is None: return # If the PCA result is None, return
        self.generate_plot(pca_result, labels, df, file_name, plot_folder) # Generate the plot

    def check_min_points(self, labels, min_points=3): # Check if there are enough points to plot
        return len(np.unique(labels)) >= min_points # Return if the number of unique labels is greater than or equal to the minimum points

    def apply_pca(self, distance_matrix): # Apply PCA
        try: # Try to apply PCA
            pca = PCA(n_components=2) # Get the PCA model
            return pca.fit_transform(SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(distance_matrix)) # Return the PCA result
        except Exception as e: # Handle the exception
            print(f"PCA error: {e}") # Print the error
            return None # Return None

    def generate_plot(self, pca_result, labels, df, file_name, plot_folder): # Generate the plot
        cluster_to_clade = self.map_cluster_to_clade(labels, df) # Map the cluster to clade
        clade_labels = [cluster_to_clade.get(cluster, 'Unknown') for cluster in labels] # Get the clade labels
        self.create_scatter_plot(pca_result, clade_labels, file_name, plot_folder) # Create the scatter plot

    def map_cluster_to_clade(self, labels, df): # Map the cluster to clade
        cluster_to_clade = {} # Dictionary of cluster to clade
        for cluster in set(labels): # Iterate over the clusters
            indices = [i for i, label in enumerate(labels) if label == cluster] # Get the indices
            clades = df.iloc[indices]['Clade'] # Get the clades
            most_common_clade = clades.mode()[0] if not clades.empty else 'Unknown' # Get the most common clade
            cluster_to_clade[cluster] = most_common_clade # Add the cluster to clade mapping
        return cluster_to_clade # Return the cluster to clade mapping


    def create_scatter_plot(self, pca_result, clade_labels, file_name, plot_folder):
        plt.figure(figsize=(10, 8))  # Set the figure size

        clade_labels = np.array(clade_labels) # Convert to numpy array
        
        unique_labels = np.unique(clade_labels) # Get the unique labels
        palette = sns.color_palette("hsv", n_colors=len(unique_labels)) # Get the color palette
        markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd'] # Lista di markers
        
        # Plot points
        for i, label in enumerate(unique_labels): # For each unique label
            idx = clade_labels == label # Get the indices
            sns.scatterplot(x=pca_result[idx, 0], y=pca_result[idx, 1], # Plot the scatter plot
                            color=palette[i], label=label,  
                            marker=markers[i % len(markers)],
                            s=50,  
                            alpha=0.8,
                            edgecolor='k', linewidth=1)
        
        plt.title(f'Clustering Results for {file_name}')  # Set Title
        plt.legend(title='Clade', bbox_to_anchor=(1.05, 1), loc='upper left')  # Set Legend
        plt.tight_layout()  # Optimizes the plot
        plt.savefig(os.path.join(plot_folder, file_name))  # Save the plot
        plt.close()  # Close Plot

        # Clear memory
        del pca_result, clade_labels  
        gc.collect()
    
    def plot_dendrogram(self, distance_matrix, clade_labels, plot_folder, file_name, max_d=1.0, max_samples=7):
        if distance_matrix.size == 0: # Check if the distance matrix is empty
            print("Distance Matrix Empty. Skip Plotting ")
            return

        if issparse(distance_matrix): # Check if the distance matrix is sparse
            distance_matrix_dense = distance_matrix.todense()
        else:
            distance_matrix_dense = distance_matrix
            
        if distance_matrix_dense.shape[0] > max_samples: # Check if the number of samples is greater than the maximum samples
            indices = np.random.choice(distance_matrix_dense.shape[0], size=max_samples, replace=False)
            sampled_matrix = distance_matrix_dense[indices][:, indices]
            sampled_clade_labels = [clade_labels[i] for i in indices]  # Campiona i clade labels
        else:
            sampled_matrix = distance_matrix_dense
            sampled_clade_labels = clade_labels

        try: # Try to calculate the linkage
            Z = linkage(sampled_matrix, 'ward')
        except ValueError as e:
            print(f"Error linkage calculate: {e}")
            return

        plt.figure(figsize=(15, 10)) # Set the figure size
        dendrogram(Z, color_threshold=max_d, labels=sampled_clade_labels) # Plot the dendrogram
        plt.title(f'Dendrogram_{file_name}') # Set the title
        plt.xlabel('Sample') # Set the x label
        plt.ylabel('Distance') # Set the y label

        plt.savefig(os.path.join(plot_folder, f"Dendrogram_{file_name}.png")) # Save the plot
        plt.close() # Close Plot
        
        # Clear memory
        del distance_matrix_dense, sampled_matrix, sampled_clade_labels, Z
        gc.collect()
