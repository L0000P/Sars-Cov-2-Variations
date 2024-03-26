import os
import logging
import shutil
import glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col, array, concat_ws

def process_dataframe(file_path, diff_temp_dir):
    print(f"Processing file {file_path}")
    df = spark.read.format('csv').option('header', 'true').option('delimiter', '\t').option('inferSchema', 'true').load(file_path)
    df = df.toDF(*(c.replace('.', '_') for c in df.columns))
    
    clade_index = df.columns.index("Country") + 6 if "sample" in df.columns else df.columns.index("Country") + 5
    mutation_columns = df.columns[clade_index + 1:]

    df = df.withColumn("Mutations", array([expr(f"CASE WHEN `{col}` = 1 THEN '{col}' END") for col in mutation_columns]))
    df_final = df.select(col("Country").alias("Geo_Location"), col(df.columns[clade_index]).alias("Clade"), "Mutations")
    df_final = df_final.withColumn("Mutations", concat_ws(",", "Mutations"))
    
    write_geo_data(df_final, diff_temp_dir)
    df.unpersist()

def write_geo_data(df, output_folder):
    countries = df.select("Geo_Location").distinct().collect()
        
    for country_row in countries:
        geo_location = country_row["Geo_Location"]
        if geo_location is None:
            continue
        if ',' in geo_location:
            geo_location = geo_location.split(',', 1)[0]
            
        df_filtered = df.filter(df["Geo_Location"] == geo_location)
        geo_location_output_path = os.path.join(output_folder, geo_location.replace(' ', '_').replace('/', '_').replace(':', '_'))
        df_filtered.write.csv(path=geo_location_output_path, mode="append", header=True)

def safe_mkdirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directory {path}: {e}")
        raise

def merge_and_clean(output_folder):
    for geo_dir in os.listdir(output_folder):
        full_path = os.path.join(output_folder, geo_dir)
        if os.path.isdir(full_path):
            csv_files = sorted(glob.glob(f"{full_path}/part*.csv"))
            if csv_files:
                with open(csv_files[0], 'r') as f:
                    header = f.readline().strip()
                with open(f"{output_folder}/{geo_dir}.csv", 'w') as merged_file:
                    merged_file.write(header + '\n')
                    for file in csv_files:
                        with open(file, 'r') as infile:
                            infile.readline()  
                            shutil.copyfileobj(infile, merged_file)
                        os.remove(file)  
            shutil.rmtree(full_path)
            print(f"Removed directory {full_path} regardless of its contents.")
        else:
            print(f"Directory {full_path} does not exist or is not a directory.")

def rename_partitions(src_dir, dst_dir):
    for subdir in os.listdir(src_dir):
        src_subdir_path = os.path.join(src_dir, subdir)
        dst_subdir_path = os.path.join(dst_dir, subdir)

        if not os.path.exists(dst_subdir_path):
            os.makedirs(dst_subdir_path)

        for file in os.listdir(src_subdir_path):
            src_file_path = os.path.join(src_subdir_path, file)
            dst_file_path = os.path.join(dst_subdir_path, file)

            if os.path.exists(dst_file_path):
                os.remove(dst_file_path)
            
            shutil.move(src_file_path, dst_file_path)

input_folder= "../matrici_ncbi_2021_2022"
output_folder = "CountriesSpark"
temp_dir = "temp"

spark = SparkSession.builder \
    .appName("divideByCountry") \
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

merge_and_clean(output_folder)
print("Operation Completed.")
spark.stop()
