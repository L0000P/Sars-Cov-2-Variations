from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col, array, concat_ws
import os
import logging
import shutil
import urllib.parse
import re
import subprocess

def process_dataframe(file_path, diff_temp_dir):
    print(f"Processing file {file_path}")
    df = spark.read.format('csv').option('header', 'true').option('delimiter', '\t').option('inferSchema', 'true').load(file_path)
    df = df.toDF(*(c.replace('.', '_') for c in df.columns))
    
    clade_index = df.columns.index("Country") + 6 if "sample" in df.columns else df.columns.index("Country") + 5
    mutation_columns = df.columns[clade_index + 1:]

    df = df.withColumn("Mutations", array([expr(f"CASE WHEN `{col}` = 1 THEN '{col}' END") for col in mutation_columns]))

    df_final = df.select("Country", "Geo_Location", col(df.columns[clade_index]).alias("Clade"), "Mutations")
    df_final = df_final.withColumn("Mutations", concat_ws(",", "Mutations"))
    df_final.repartition("Country").write.partitionBy("Country").option("header", "true").format("csv").mode("overwrite").save(diff_temp_dir)
    df.unpersist()

def safe_mkdirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directory {path}: {e}")
        raise

def rename_partitions(src_dir, dst_dir):
    for subdir in os.listdir(src_dir):
        if subdir.startswith('Country='):
            # Decodifica URL per gestire correttamente i caratteri speciali come %3A
            country_encoded = subdir.replace('Country=', '')
            country_decoded = urllib.parse.unquote(country_encoded)
            
            # Sostituisci la virgola e altri caratteri non desiderati con underscore
            # e rimuovi gli spazi extra
            country_sanitized = re.sub(r'[^\w\s]', '_', country_decoded)
            country_sanitized = re.sub(r'\s+', '_', country_sanitized)
            
            country_path = os.path.join(dst_dir, country_sanitized)
            safe_mkdirs(country_path)
            
            for file in os.listdir(os.path.join(src_dir, subdir)):
                src_file_path = os.path.join(src_dir, subdir, file)
                dst_file_path = os.path.join(country_path, file)
                shutil.move(src_file_path, dst_file_path)

def cleanup_temp_dir(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
def merge_and_clean(cartella_output):
    for country_dir in os.listdir(cartella_output):
        print(f"Merge {country_dir}...")
        full_path = os.path.join(cartella_output, country_dir)
        if os.path.isdir(full_path):
            cmd_merge = f'copy /b "{full_path}\\part*.csv" "{cartella_output}\\{country_dir}.csv"'
            subprocess.call(cmd_merge, shell=True)
            cmd_del = f'del "{full_path}\\part*.csv"'
            subprocess.call(cmd_del, shell=True)
            shutil.rmtree(full_path)

cartella_input = "matrici_ncbi_2021_2022"
cartella_output = "Countries"
temp_dir = "temp"

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

os.makedirs(cartella_output, exist_ok=True)


for file_name in os.listdir(cartella_input):
    spark.catalog.clearCache()
    
    file_path = os.path.join(cartella_input, file_name)
    diff_temp_dir = os.path.join(temp_dir, file_name)

    process_dataframe(file_path, diff_temp_dir)
    rename_partitions(diff_temp_dir, cartella_output)

cleanup_temp_dir(temp_dir)
# After processing all files, merge CSV files in each country directory
merge_and_clean(cartella_output)

print("Operazione completata.")
spark.stop()