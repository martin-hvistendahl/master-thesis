# Databricks notebook source
df_all_liftlog = (
    spark.read.csv("/Volumes/craneds_dev/maxedge_lhdp/crane_files/csv_categories/liftlog/*101**.csv", header=True, inferSchema=True)
)
display(
    df_all_liftlog.groupBy("movementtype").count().orderBy("count", ascending=False)
)

display(
    df_all_liftlog.dropDuplicates(["movementid"])
    .groupBy("movementtype")
    .count()
    .orderBy("count", ascending=False)
)
import os
import pandas as pd
from scipy.spatial.distance import euclidean
from pyspark.sql import functions as F

# -----------------------------
# Config
# -----------------------------
liftlog_dir = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/csv_categories/liftlog/"
filename_filter = "101"
min_rows = 100  # skip very small files; adjust if needed

# Assumes df_all_liftlog already exists
# If not, create/load it before running this script.

# -----------------------------
# List matching files
# -----------------------------
all_files = dbutils.fs.ls(liftlog_dir)
filepaths = [
    f.path for f in all_files
    if filename_filter in f.name and f.name.endswith(".csv")
]

print(f"Found {len(filepaths)} matching files")

if not filepaths:
    raise ValueError("No matching CSV files found.")

# -----------------------------
# Full dataset distribution
# -----------------------------
full_dist_pdf = (
    df_all_liftlog
    .select("movementtype")
    .where(F.col("movementtype").isNotNull())
    .groupBy("movementtype")
    .count()
    .orderBy("movementtype")
    .toPandas()
)

if full_dist_pdf.empty:
    raise ValueError("Full dataset has no non-null movementtype values.")

full_dist = full_dist_pdf.set_index("movementtype")["count"]
full_dist_norm = full_dist / full_dist.sum()

print("\nFull dataset movementtype distribution:")
print(full_dist_norm)

# -----------------------------
# Score each file
# -----------------------------
results = []

for fp in filepaths:
    try:
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(fp)
            .select("movementtype")
        )

        total_rows = df.count()
        non_null_rows = df.where(F.col("movementtype").isNotNull()).count()

        if non_null_rows == 0:
            print(f"Skipping {fp}: no non-null movementtype values")
            continue

        if total_rows < min_rows:
            print(f"Skipping {fp}: only {total_rows} rows (< {min_rows})")
            continue

        dist_pdf = (
            df.where(F.col("movementtype").isNotNull())
            .groupBy("movementtype")
            .count()
            .orderBy("movementtype")
            .toPandas()
        )

        dist = dist_pdf.set_index("movementtype")["count"]
        dist = dist.reindex(full_dist.index, fill_value=0)
        dist_norm = dist / dist.sum()

        distance = euclidean(full_dist_norm.values, dist_norm.values)

        results.append({
            "filepath": fp,
            "filename": os.path.basename(fp),
            "total_rows": total_rows,
            "non_null_movementtype_rows": non_null_rows,
            "distance": distance
        })

    except Exception as e:
        print(f"Error processing {fp}: {e}")

if not results:
    raise ValueError("No valid files were scored.")

# -----------------------------
# Rank results
# -----------------------------
results_df = pd.DataFrame(results).sort_values("distance").reset_index(drop=True)

print("\nTop 10 closest files:")
print(results_df.head(10).to_string(index=False))

# -----------------------------
# Best file
# -----------------------------
best_file = results_df.iloc[0]["filepath"]
best_filename = results_df.iloc[0]["filename"]
min_dist = results_df.iloc[0]["distance"]

print("\nBest representative file:")
print(best_file)
print(f"Euclidean distance: {min_dist:.6f}")

# -----------------------------
# Detailed comparison for best file
# -----------------------------
df_best = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(best_file)
    .select("movementtype")
)

best_dist_pdf = (
    df_best
    .where(F.col("movementtype").isNotNull())
    .groupBy("movementtype")
    .count()
    .orderBy("movementtype")
    .toPandas()
)

best_dist = best_dist_pdf.set_index("movementtype")["count"]
best_dist = best_dist.reindex(full_dist.index, fill_value=0)
best_dist_norm = best_dist / best_dist.sum()

comparison = (
    full_dist_norm.rename("full_share").to_frame()
    .join(best_dist_norm.rename("best_file_share"))
)

comparison["abs_diff"] = (comparison["full_share"] - comparison["best_file_share"]).abs()
comparison = comparison.sort_values("abs_diff", ascending=False)

print("\nLargest movementtype share differences (best file vs full dataset):")
print(comparison.head(10).to_string())

# Optional: keep the results in Spark / pandas objects for later use
best_representative_file = best_file
best_representative_summary = results_df.head(10)
best_file_distribution_comparison = comparison

# COMMAND ----------

from pyspark.sql import functions as F


# IMPORT liftlog and alarms
df_all_liftlog = spark.read.csv("/Volumes/craneds_dev/maxedge_lhdp/crane_files/csv_categories/liftlog/*101**.csv", header=True, inferSchema=True)
print(f"Number of liftlogs read: {df_all_liftlog.count()}")
dupes_count_liftlog = df_all_liftlog.count() - df_all_liftlog.dropDuplicates().count()
print(f"Number of duplicate rows deleted from liftlog: {dupes_count_liftlog}")
df_all_liftlog = (
    df_all_liftlog
    # Build start timestamp directly
    .withColumn("lift_start_ts", F.col("started"))
    
    # Build end timestamp using movementduration (seconds)
    .withColumn(
        "lift_end_ts",
        F.expr("timestampadd(SECOND, movementduration, started)")
    )
    
    # Define lift_id (use movementid if that is unique per lift)
    .withColumn("lift_id", F.col("movementid"))
    
    # Reorder columns so important ones are first
    .select(
        "lift_id",
        "lift_start_ts",
        "lift_end_ts",
        *[c for c in df_all_liftlog.columns if c not in {"started", "_source_file"}]
    )
)

df_all_alarms = spark.read.csv("/Volumes/craneds_dev/maxedge_lhdp/crane_files/csv_categories/alarms/*101**.csv", header=True, inferSchema=True)
print(f"Number of alarms read: {df_all_alarms.count()}")

dupes_count_alarms = df_all_alarms.count() - df_all_alarms.dropDuplicates().count()
print(f"Number of duplicate rows deleted from alarms: {dupes_count_alarms}")


df_all_liftlog = df_all_liftlog.dropDuplicates()
df_all_alarms = df_all_alarms.dropDuplicates()
print(f"Number of liftlogs without duplicates: {df_all_liftlog.count()}")
print(f"Number of alarms without duplicates: {df_all_alarms.count()}")


# List of alarm ids to filter out
alarm_ids_to_remove = [1335, 1336, 1285, 1275, 1288, 1276, 1286, 1287]

# Count and show how many of each alarm id is deleted and the text for it
deleted_alarms = df_all_alarms.filter(df_all_alarms["id"].isin(alarm_ids_to_remove))
deleted_counts = deleted_alarms.groupBy("id", "text").agg(F.count("*").alias("count_deleted"))
display(deleted_counts)

df_all_alarms = df_all_alarms.filter(~df_all_alarms["id"].isin(alarm_ids_to_remove))

df_alarms_with_lift = (
    df_all_alarms.alias("a")
    .join(
        df_all_liftlog.select("lift_id", "lift_start_ts", "lift_end_ts").alias("l"),
        (F.col("a.timestamp_utc") >= F.col("l.lift_start_ts")) &
        (F.col("a.timestamp_utc") <= F.col("l.lift_end_ts")),
        how="left"
    )
    .select("a.*", F.col("l.lift_id"))
    .drop("_source_file")
)

alarm_counts = (
    df_alarms_with_lift.groupBy("lift_id")
    .agg(F.count("*").alias("alarm_count"))
)

df_all_liftlog = (
    df_all_liftlog
    .join(alarm_counts, on="lift_id", how="left")
    .fillna({"alarm_count": 0})
)
df_alarms_with_lift.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("all_alarms")
df_all_liftlog.write.mode("overwrite").option("mergeSchema", "true").saveAsTable("all_liftlog")


# COMMAND ----------

df_all_raw = spark.read.csv("/Volumes/craneds_dev/maxedge_lhdp/crane_files/csv_categories/raw/*101**.csv", header=True, inferSchema=True)
print(f"Number of liftlogs read: {df_all_raw.count()}")

from pyspark.sql.functions import coalesce, col, when

df_all_raw = (
    df_all_raw
    .filter(coalesce(col("bool_val"), col("int_val"), col("float_val")).isNotNull())
    .select(
        "timestamp_utc",
        "tagname",
        coalesce(col("bool_val"), col("int_val"), col("float_val")).cast("double").alias("value"),
        when(col("bool_val").isNotNull(), "bool")
        .when(col("int_val").isNotNull(), "int")
        .when(col("float_val").isNotNull(), "float")
        .otherwise(None).alias("datatype")
    )
)

# Join with df_all_liftlog on timestamp_utc between lift_start_ts and lift_end_ts
df_all_raw_with_lift = (
    df_all_raw.alias("c")
    .join(
        df_all_liftlog.select("lift_id", "lift_start_ts", "lift_end_ts").alias("l"),
        (col("c.timestamp_utc") >= col("l.lift_start_ts")) & (col("c.timestamp_utc") <= col("l.lift_end_ts")),
        how="inner"
    )
    .select("c.*", col("l.lift_id"))
)

df_all_raw_with_lift.printSchema()
display(df_all_raw_with_lift.limit(5))


# COMMAND ----------

df_all_raw_with_lift.write.format("delta").mode("overwrite").saveAsTable("default.df_all_raw_liftid")
spark.sql("SHOW DATABASES").show(truncate=False)