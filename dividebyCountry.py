from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace
import os
import logging
import shutil
import urllib.parse
import pandas as pd
import re

def safe_mkdirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directory {path}: {e}")
        raise

def rename_partitions(src_dir, dst_dir):
    for subdir in os.listdir(src_dir):
        if subdir.startswith('Country='):
            country = urllib.parse.unquote(subdir.replace('Country=', ''))
            # Use regular expression to clean special characters
            # This regex replaces anything that is not an alphanumeric character or an underscore with an underscore
            country = re.sub(r'[^\w\s]', '_', country)
            # Replace spaces with underscores for consistency
            country = country.replace(' ', '_')
            country_path = os.path.join(dst_dir, country)
            safe_mkdirs(country_path)
            for file in os.listdir(os.path.join(src_dir, subdir)):
                shutil.move(os.path.join(src_dir, subdir, file), os.path.join(country_path, file))

def cleanup_temp_dir(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
def merge_and_clean(cartella_output):
    for country_dir in os.listdir(cartella_output):
        print(f"Merge {country_dir}...")
        merged_df = pd.DataFrame()
        full_path = os.path.join(cartella_output, country_dir)
        if os.path.isdir(full_path):
            all_files = [os.path.join(full_path, f) for f in os.listdir(full_path) if os.path.isfile(os.path.join(full_path, f)) and f.endswith('.csv')]
            if all_files:
                for file in all_files:
                    df = pd.read_csv(file, low_memory=False)
                    merged_df = pd.concat([merged_df, df], ignore_index=True)
                    os.remove(file)
                merged_file_path = os.path.join(cartella_output, f"{country_dir}.csv")
                merged_df.to_csv(merged_file_path, index=False)
                shutil.rmtree(full_path)


cartella_input = "matrici_ncbi_2021_2022"
cartella_output = "Countries"
temp_dir = "temp"

os.makedirs(cartella_output, exist_ok=True)

logging.getLogger("py4j").setLevel(logging.ERROR)

spark = SparkSession.builder \
    .appName("dividebyCountry") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.instances", "2") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "300") \
    .config("spark.sql.debug.maxToStringFields", 200) \
    .config("spark.broadcast.blockSize", "10m") \
    .getOrCreate()

for file_name in os.listdir(cartella_input):
    file_path = os.path.join(cartella_input, file_name)
    diff_temp_dir = os.path.join(temp_dir, file_name)  # Use file_name to create a unique temp directory for each file
    
    if os.path.exists(diff_temp_dir):
        print(f"Directory {diff_temp_dir} already exists. Skipping file {file_name}.")
        continue 
    
    print(f"Processing file {file_path}")
    df = spark.read.format('csv').option('header', 'true').option('delimiter', '\t').option('inferSchema', 'true').load(file_path)
    df = df.withColumn("Country", regexp_replace(col("Country"), "\\.", "_"))
    df.repartition("Country").write.partitionBy("Country").option("header", "true").mode("append").format("csv").save(diff_temp_dir)
    df.unpersist()
    rename_partitions(diff_temp_dir, cartella_output)
    cleanup_temp_dir(diff_temp_dir)  # Cleanup after the operation

# After processing all files, merge CSV files in each country directory
merge_and_clean(cartella_output)

print("Operazione completata.")
spark.stop()
