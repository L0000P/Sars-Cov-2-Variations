import pandas as pd
import os
import shutil
import re

def process_dataframe(file_path, output_dir):   # Process dataframe
    print(f"Processing file {file_path}")
    df = pd.read_csv(file_path, delimiter='\t', header=0, low_memory=False) # Read csv file
    
    clade_index = df.columns.get_loc("clade")   # Get index of clade column and mutation columns
    mutation_columns = df.columns[clade_index + 1:] # Get all columns after clade column (mutation columns)
    
    df['Mutations'] = df.apply(lambda row: ','.join([col for col in mutation_columns if row[col] == 1]), axis=1) # Calculate mutations
    
    df['Geo_Location'] = df['Geo_Location'].apply(lambda x: x.split(',')[0] if pd.notnull(x) else x) # Truncate Geo_Location
    
    df = df[["Country", "Geo_Location", df.columns[clade_index], "Mutations"]]    # Select relevant columns
    
    for country, group in df.groupby("Country"):  # Group by country
        country_sanitized = re.sub(r'[^\w\s]', '_', country)
        country_sanitized = re.sub(r'\s+', '_', country_sanitized)
        
        country_path = os.path.join(output_dir, f"{country_sanitized}.csv") # Create path for country
        
        if os.path.exists(country_path):    # Write to file
            group.to_csv(country_path, mode='a', header=False, index=False)     # Append to file
        else:
            group.to_csv(country_path, mode='w', header=True, index=False)      # Create new file


def cleanup_dir(dir_path):  # Clean up directory
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    
input_folder = "../matrici_ncbi_2021_2022"
output_folder = "CountriesPD"

cleanup_dir(output_folder)

for file_name in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file_name)
    process_dataframe(file_path, output_folder)

print("Operation Completed.")