from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col, array, concat_ws
import os
import logging
import shutil
import urllib.parse
import re
import subprocess

def process_dataframe(file_path, diff_temp_dir):    # Process dataframe and save it to a temporary directory
    print(f"Processing file {file_path}")
    df = spark.read.format('csv').option('header', 'true').option('delimiter', '\t').option('inferSchema', 'true').load(file_path) # Load CSV file
    df = df.toDF(*(c.replace('.', '_') for c in df.columns))    # Replace dots in column names with underscores
    
    clade_index = df.columns.index("Country") + 6 if "sample" in df.columns else df.columns.index("Country") + 5    # Find the index of the column containing the clade
    mutation_columns = df.columns[clade_index + 1:] # Get the list of mutation columns

    df = df.withColumn("Mutations", array([expr(f"CASE WHEN `{col}` = 1 THEN '{col}' END") for col in mutation_columns])) # Create a new column containing an array of mutations

    df_final = df.select("Country", "Geo_Location", col(df.columns[clade_index]).alias("Clade"), "Mutations") # Select only the columns we need
    df_final = df_final.withColumn("Mutations", concat_ws(",", "Mutations")) # Concatenate the array of mutations into a string
    df_final.repartition("Country").write.partitionBy("Country").option("header", "true").format("csv").mode("overwrite").save(diff_temp_dir) # Save the dataframe partitioned by country
    df.unpersist() # Unpersist the dataframe to free memory

def safe_mkdirs(path): # Create a directory if it does not exist
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directory {path}: {e}")
        raise

def rename_partitions(src_dir, dst_dir): # Rename the partitions to use the country name
    for subdir in os.listdir(src_dir):
        if subdir.startswith('Country='):
            country_encoded = subdir.replace('Country=', '')
            country_decoded = urllib.parse.unquote(country_encoded)
            country_sanitized = re.sub(r'[^\w\s]', '_', country_decoded)
            country_sanitized = re.sub(r'\s+', '_', country_sanitized)
            
            country_path = os.path.join(dst_dir, country_sanitized)
            safe_mkdirs(country_path)
            
            for file in os.listdir(os.path.join(src_dir, subdir)):
                src_file_path = os.path.join(src_dir, subdir, file)
                dst_file_path = os.path.join(country_path, file)
                shutil.move(src_file_path, dst_file_path)


def cleanup_temp_dir(temp_dir): # Clean up the temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
def merge_and_clean(output_folder): # Merge CSV files in each country directory and clean up the temporary directories
    for country_dir in os.listdir(output_folder):
        print(f"Merge {country_dir}...")
        full_path = os.path.join(output_folder, country_dir)
        if os.path.isdir(full_path):
            cmd_merge = f'copy /b "{full_path}\\part*.csv" "{output_folder}\\{country_dir}.csv"'
            subprocess.call(cmd_merge, shell=True)
            cmd_del = f'del "{full_path}\\part*.csv"'
            subprocess.call(cmd_del, shell=True)
            shutil.rmtree(full_path)

input_folder= "../matrici_ncbi_2021_2022"
output_folder = "../Countries"
temp_dir = "../temp"

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

spark = SparkSession.builder \
    .appName("dividebyCountry") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "300") \
    .config("spark.sql.debug.maxToStringFields", 200) \
    .config("spark.broadcast.blockSize", "10m") \
    .config("spark.local.dir", temp_dir) \
    .getOrCreate()

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    spark.catalog.clearCache()
    
    file_path = os.path.join(input_folder, file_name)
    diff_temp_dir = os.path.join(temp_dir, file_name)

    process_dataframe(file_path, diff_temp_dir)
    rename_partitions(diff_temp_dir, output_folder)

cleanup_temp_dir(temp_dir)
# After processing all files, merge CSV files in each country directory
merge_and_clean(output_folder)

print("Operazione completata.")
spark.stop()