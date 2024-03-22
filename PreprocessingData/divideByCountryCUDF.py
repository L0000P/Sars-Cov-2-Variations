import cudf
import os
import re
import pandas as pd 

def truncate_country_values(file_path): # Truncate the Geo_Location values to only keep the country name
    df = pd.read_csv(file_path) # Read the file
    df['Geo_Location'] = df['Geo_Location'].apply(lambda x: x.split(',')[0] if isinstance(x, str) and ',' in x else x) # Split the Geo_Location values by ',' and keep only the first part
    df.to_csv(file_path, index=False) # Save the file

def process_file(file_path, output_dir): # Process the file
    print(f"Processing file: {file_path}")
    df = cudf.read_csv(file_path, delimiter='\t', header=0) # Read the file
    start_index = df.columns.get_loc('Name') + 1 # Get the index of the column 'Name' and add 1 to get the start index
    for i in range(start_index, len(df.columns)-1): # Iterate over the columns
        df[df.columns[i - start_index]] = df[df.columns[i]] # Rename the columns
    df = df.iloc[:, :-(start_index)] # Remove the last columns
    
    clade_index = df.columns.get_loc('clade') # Get the index of the column 'clade'
    
    mutation_columns = df.columns[clade_index + 1:] # Get the mutation columns
    
    df['Mutations'] = '' # Create a new column 'Mutations'
    for col in mutation_columns: # Iterate over the mutation columns
        condition = df[col] == 1 # Get the rows where the value is 1
        df['Mutations'] = df['Mutations'].where(~condition, df['Mutations'] + col + ',') # Add the mutation to the 'Mutations' column if the condition is met
    df['Mutations'] = df['Mutations'].str.rstrip(',') # Remove the trailing comma

    df = df[['Country', 'Geo_Location', 'clade', 'Mutations']] # Keep only the relevant columns
    
    for country, group in df.groupby('Country'): # Iterate over the groups
        country_sanitized = re.sub(r'[^\w\s]', '_', country) # Sanitize the country name
        country_path = os.path.join(output_dir, f"{country_sanitized}.csv") # Create the output file path
        if os.path.exists(country_path): # Check if the file exists
            group.to_pandas().to_csv(country_path, mode="a", index=False, header=False) # Append the group to the file
            truncate_country_values(country_path) # Truncate the Geo_Location values
        else:
            with open(country_path, 'w') as f: # Create the file
                f.write("Country,Geo_Location,Clade,Mutations\n") # Write the header
            group.to_pandas().to_csv(country_path, mode="a", index=False, header=False) # Append the group to the file
            truncate_country_values(country_path) # Truncate the Geo_Location values

def process_folder(input_folder, output_dir): # Process the folder
    os.makedirs(output_dir, exist_ok=True) # Create the output directory
    for file_name in os.listdir(input_folder): # Iterate over the files in the input folder
        file_path = os.path.join(input_folder, file_name) # Get the file path
        if os.path.isfile(file_path): # Check if the path is a file
            process_file(file_path, output_dir) # Process the file

input_folder = "../matrici_ncbi_2021_2022" # Input folder 
output_dir = "CountriesCUDF" # Output folder

process_folder(input_folder, output_dir) # Process the folder

print("Operation Completed!")
