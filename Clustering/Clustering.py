from Clusterizer import Clusterizer
import os

def main():
    input_folder = "../Countries" # Folder with the csv files
    types_clustering = ["km", "ka", "gc", "db", "op", "hdb"] # Types of clustering to be used

    for clt_type in types_clustering: # For each type of clustering
        plot_folder = f"../Plots/{clt_type}" # Folder to save the plots
        os.makedirs(plot_folder, exist_ok=True) # Create the folder if it doesn't exist
        clt = Clusterizer(clt_type) # Create the clusterizer object
        for file_name in os.listdir(input_folder): # For each file in the input folder
            if file_name.endswith('.csv'): # If the file is a csv file
                clt.process_file(input_folder, plot_folder, file_name) # Process the file

if __name__ == "__main__": # Run the main function
    main()