# Databricks notebook source
import zipfile, os
crane_files_dir = "/Volumes/craneds_dev/maxedge_lhdp/crane_files"
print(f"crane_files_dir: {crane_files_dir}")
common_extract_path = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/unzipped_all"

os.makedirs(common_extract_path, exist_ok=True)

for file_name in os.listdir(crane_files_dir):
    file_path = os.path.join(crane_files_dir, file_name)
    if file_name.lower().endswith(".zip") and os.path.isfile(file_path):
        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(common_extract_path)



# COMMAND ----------

category_folder = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/csv_categories"
if not os.path.exists(category_folder):
    os.makedirs(category_folder)

csv_suffixes = [
    "_alarms.csv",
    "_annotations.csv",
    "_eventlog.csv",
    "_liftlog.csv",
    "_raw.csv"
]

for file_name in os.listdir(common_extract_path):
    for suffix in csv_suffixes:
        if file_name.lower().endswith(suffix):
            src_path = os.path.join(common_extract_path, file_name)
            dest_path = os.path.join(category_folder, file_name)
            os.rename(src_path, dest_path)
            break


# COMMAND ----------



for suffix in csv_suffixes:
    folder_name = suffix.replace(".csv", "").replace("_", "")
    folder_path = os.path.join(category_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
category_folder = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/csv_categories"

for file_name in os.listdir(category_folder):
    for suffix in csv_suffixes:
        if file_name.lower().endswith(suffix):
            src_path = os.path.join(category_folder, file_name)
            folder_name = suffix.replace(".csv", "").replace("_", "")
            dest_folder = os.path.join(category_folder, folder_name)
            os.makedirs(dest_folder, exist_ok=True)
            dest_path = os.path.join(dest_folder, file_name)
            os.rename(src_path, dest_path)
            break