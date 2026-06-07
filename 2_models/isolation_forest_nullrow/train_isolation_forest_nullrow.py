# Databricks notebook source
# MAGIC %md
# MAGIC Here are the databases load so computation is already done with refering raw data with an lift id

# COMMAND ----------

from pyspark.sql import functions as F

df_raw_with_lift = spark.table("default.df_all_raw_liftid")
df_alarms_with_lift_fit = spark.table("default.df_alarms_fit")
df_liftlog = spark.table("default.all_liftlog")
df_alarms_with_lift = spark.table("default.df_alarms")


df_liftlog_5 = df_liftlog.filter(F.col("movementtype") == 5)
df_liftlog = df_liftlog.filter(F.col("movementtype") != 5)



# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import when
from itertools import chain
import pandas as pd
import numpy as np


# ============================================================
# FULL UPDATED PREPROCESSING SCRIPT
# ============================================================
# This script:
#   1. Defines new tag groups
#   2. Assigns tag_group
#   3. Applies group-specific bucket sizes
#   4. Creates lift_group_bucket
#   5. Pivots each group to a wide dataframe
#   6. Optionally prints NULL percentages
#   7. Removes rows for lift_ids found in df_liftlog_5
#   8. Fills NULLs using rule-based zero filling and
#      distance-weighted temporal interpolation within each lift_id
#
# Required input dataframes:
#   - df_raw_with_lift
#   - df_liftlog_5
#
# Output:
#   - df_pivoted_dict   : dict of Spark DataFrames
#   - df_filled_dict    : dict of Pandas DataFrames
# ============================================================

TIMESTAMP_COL = "timestamp_utc"   # change if needed

# ============================================================
# 1) NEW TAG GROUPS
# ============================================================

# 1) Always-on / baseline signals
tagnames_always_on = [
    "ImportS7Device.Drives.MainRectifier.DCLinkVoltage",
    "ImportS7Device.General.WindSpeed",
    "ImportS7Device.SafetySystems.AOPS RopeForce",
    "ImportS7Device.CBM.SlewBearing_Moment",
    "ImportS7Device.General.PedestalMoment",
    "ImportS7Device.CBM.LuffingWinch_Moment",
    "ImportS7Device.CBM.MainHoistWinch_Moment",
    "ImportS7Device.General.BoomLoad",
    "ImportS7Device.General.HoistLoadPctSWL",
]

tagnames_boom_active = [
    "ImportS7Device.Encoders.BoomA_Raw",
    "ImportS7Device.Encoders.BoomB_Raw",
    "ImportS7Device.General.BoomAngle",
    "ImportS7Device.General.BoomRadius",
    "ImportS7Device.Drives.Boom_Status.ActualSpeed",
    "ImportS7Device.Drives.Boom_Status.ActualPower",
    "ImportS7Device.Drives.Boom_Control.SpeedReference",
    "ImportS7Device.Joysticks.Boom",
    "ImportS7Device.Drives.Boom_Status.ActualCurrent",
    "ImportS7Device.Drives.Boom_Status.ActualTorque",
    "ImportS7Device.Hyd.BoomPrimaryPressure",
]

tagnames_hoist_active = [
    "ImportS7Device.Drives.Hoist_Status.ActualCurrent",
    "ImportS7Device.Joysticks.Hoist",
    "ImportS7Device.Drives.Hoist_Status.ActualSpeed",
    "ImportS7Device.Drives.Hoist_Control.SpeedReference",
    "ImportS7Device.Encoders.MainA_Raw",
    "ImportS7Device.Encoders.MainB_Raw",
    "ImportS7Device.Drives.Hoist_Status.ActualPower",
    "ImportS7Device.General.HoistPosition",
]

tagnames_hyd_common = [
    "ImportS7Device.Hyd.BrakeSystemPressure",
    "ImportS7Device.Hyd.BrakeAccumulatorPressure",
]

tagnames_slew_active = [
    "ImportS7Device.Encoders.Slew_Raw",
    "ImportS7Device.General.SlewAngle",
    "ImportS7Device.Joysticks.Slew",
    "ImportS7Device.Drives.SlewA_Control.SpeedReference",
    "ImportS7Device.Drives.SlewB_Control.SpeedReference",
    "ImportS7Device.Drives.SlewA_Status.ActualSpeed",
    "ImportS7Device.Drives.SlewB_Status.ActualSpeed",
    "ImportS7Device.Drives.SlewB_Status.ActualTorque",
    "ImportS7Device.Drives.SlewB_Status.ActualCurrent",
    "ImportS7Device.Drives.SlewB_Status.DroopFeedbackSpeedReduction",
    "ImportS7Device.Drives.SlewA_Status.ActualTorque",
    "ImportS7Device.Drives.SlewA_Status.DroopFeedbackSpeedReduction",
    "ImportS7Device.Drives.SlewA_Status.ActualCurrent",
    "ImportS7Device.CBM.SlewGear_Moment",
    "ImportS7Device.General.SlewMoment",
    "ImportS7Device.Hyd.SlewBrakePressureA",
    "ImportS7Device.Hyd.SlewBrakePressureB",
]

tagnames_slew_power = [
    "ImportS7Device.Drives.SlewA_Status.ActualPower",
    "ImportS7Device.Drives.SlewB_Status.ActualPower",
    "ImportS7Device.Drives.SlewB_Status.TorqueLimEffective",
    "ImportS7Device.Drives.SlewA_Status.TorqueLimEffective",
]

tagnames_thermal_drive = [
    "ImportS7Device.Drives.Hoist_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.SlewB_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.Boom_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.SlewA_Status.DepletionLayerMaxTemp",
]

tagnames_thermal_motor = [
    "ImportS7Device.Drives.Hoist_Status.MotorTemp",
    "ImportS7Device.Drives.Boom_Status.MotorTemp",
    "ImportS7Device.Drives.SlewA_Status.MotorTemp",
    "ImportS7Device.Drives.SlewB_Status.MotorTemp",
]

tagnames_thermal_resistor = [
    "ImportS7Device.HVAC.ResistorCoolant.BrakeResistorTempA",
    "ImportS7Device.HVAC.ResistorCoolant.BrakeResistorTempB",
    "ImportS7Device.HVAC.ResistorCoolant.ResistorTankCoolantTemp",
]

# ============================================================
# 2) ALL SELECTED TAGS
# ============================================================

tagnames = (
    tagnames_always_on +
    tagnames_boom_active +
    tagnames_hoist_active +
    tagnames_hyd_common +
    tagnames_slew_active +
    tagnames_slew_power +
    tagnames_thermal_drive +
    tagnames_thermal_motor +
    tagnames_thermal_resistor
)

# ============================================================
# 3) TAG GROUP MAPPING
# ============================================================

tag_group_expr = (
    when(F.col("tagname").isin(tagnames_always_on), "always_on")
    .when(F.col("tagname").isin(tagnames_boom_active), "boom_active")
    .when(F.col("tagname").isin(tagnames_hoist_active), "hoist_active")
    .when(F.col("tagname").isin(tagnames_hyd_common), "hyd_common")
    .when(F.col("tagname").isin(tagnames_slew_active), "slew_active")
    .when(F.col("tagname").isin(tagnames_slew_power), "slew_power")
    .when(F.col("tagname").isin(tagnames_thermal_drive), "thermal_drive")
    .when(F.col("tagname").isin(tagnames_thermal_motor), "thermal_motor")
    .when(F.col("tagname").isin(tagnames_thermal_resistor), "thermal_resistor")
    .otherwise("other")
)

# ============================================================
# 4) FILTER RAW DATA AND ASSIGN TAG GROUP
# ============================================================

df_raw = (
    df_raw_with_lift
    .filter(F.col("tagname").isin(tagnames))
    .withColumn("tag_group", tag_group_expr)
    .withColumn(TIMESTAMP_COL, F.col(TIMESTAMP_COL).cast("timestamp"))
)

# ============================================================
# 5) BUCKET SIZE PER GROUP (SECONDS)
# ============================================================

bucket_sizes = {
    "always_on": 5,
    "boom_active": 5,
    "hoist_active": 5,
    "hyd_common": 10,
    "slew_active": 5,
    "slew_power": 60,
    "thermal_drive": 30,
    "thermal_motor": 300,
    "thermal_resistor": 120,
}

bucket_map_expr = F.create_map([F.lit(x) for x in chain(*bucket_sizes.items())])

# ============================================================
# 6) APPLY GROUP-SPECIFIC BUCKETING
# ============================================================

df_raw_buckets = (
    df_raw
    .withColumn("bucket_size_s", bucket_map_expr[F.col("tag_group")].cast("int"))
    .withColumn("ts_unix", F.unix_timestamp(F.col(TIMESTAMP_COL)))
    .withColumn(
        "bucket_start_unix",
        (F.floor(F.col("ts_unix") / F.col("bucket_size_s")) * F.col("bucket_size_s")).cast("long")
    )
    .withColumn("bucket_start_ts", F.from_unixtime(F.col("bucket_start_unix")).cast("timestamp"))
    .withColumn(
        "lift_group_bucket",
        F.concat_ws("_", F.col("lift_id").cast("string"), F.col("bucket_start_unix").cast("string"))
    )
)

# ============================================================
# 7) TAG GROUPS DICTIONARY FOR PIVOTING
# ============================================================

taggroups = {
    "always_on": tagnames_always_on,
    "boom_active": tagnames_boom_active,
    "hoist_active": tagnames_hoist_active,
    "hyd_common": tagnames_hyd_common,
    "slew_active": tagnames_slew_active,
    "slew_power": tagnames_slew_power,
    "thermal_drive": tagnames_thermal_drive,
    "thermal_motor": tagnames_thermal_motor,
    "thermal_resistor": tagnames_thermal_resistor,
}

# ============================================================
# 8) PIVOT EACH GROUP INTO ITS OWN WIDE DATAFRAME
# ============================================================

df_pivoted_dict = {}

for group, tags in taggroups.items():
    df_group = df_raw_buckets.filter(F.col("tag_group") == group)

    if df_group.limit(1).count() > 0:
        df_pivoted_dict[group] = (
            df_group
            .groupBy("lift_group_bucket", "bucket_start_unix")
            .pivot("tagname", tags)
            .agg(F.mean("value"))
            .orderBy("lift_group_bucket")
        )

# ============================================================
# 9) OPTIONAL: PRINT NULL PERCENTAGE PER COLUMN
# ============================================================

# for group, df in df_pivoted_dict.items():
#     total_rows = df.count()
#     null_counts = df.select([
#         F.count(F.when(F.col(c).isNull(), c)).alias(c)
#         for c in df.columns if c not in ["lift_group_bucket", "bucket_start_unix"]
#     ]).collect()[0].asDict()
#
#     print(f"\nGroup: {group}")
#     for tag in [c for c in df.columns if c not in ["lift_group_bucket", "bucket_start_unix"]]:
#         nulls = null_counts[tag]
#         pct_null = 100 * nulls / total_rows if total_rows > 0 else 0
#         print(f"  {tag}: {nulls}/{total_rows} ({pct_null:.1f}% NULL data)")

# ============================================================
# 10) REMOVE ALL lift_id ROWS FOUND IN df_liftlog_5
# ============================================================

lift_ids_to_remove = [
    str(row["lift_id"])
    for row in df_liftlog_5.select("lift_id").distinct().collect()
]

if lift_ids_to_remove:
    pattern = f"^({'|'.join(lift_ids_to_remove)})_"
    for group in df_pivoted_dict:
        df_pivoted_dict[group] = df_pivoted_dict[group].filter(
            ~F.col("lift_group_bucket").rlike(pattern)
        )

# ============================================================
# 11) OPTIONAL: ACCESS EXAMPLES
# ============================================================

print("Available groups in df_pivoted_dict:")
for k in df_pivoted_dict.keys():
    print(" -", k)


# ============================================================
# 12) KEEP NULLS - NO ZERO FILL / NO INTERPOLATION
# ============================================================

df_null_dict = {}

for group, spark_df in df_pivoted_dict.items():
    print(f"\nKeeping NULLs for group: {group}")

    pdf = spark_df.toPandas()

    pdf = pdf.sort_values(["lift_group_bucket", "bucket_start_unix"]).copy()
    pdf = pdf.set_index("lift_group_bucket", drop=True)

    df_null_dict[group] = pdf

# Optional check
for group, pdf in df_null_dict.items():
    print(f"\n--- {group} ---")
    print(pdf.head())
    print("\nTop NULL percentages:")
    print(pdf.isna().mean().sort_values(ascending=False).head(10))
    break

# Example access:
# display(spark.createDataFrame(df_null_dict["always_on"].reset_index()))

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os

# ============================================================
# CONFIG
# ============================================================

bucket_sizes = {
    "always_on": 5,
    "boom_active": 5,
    "hoist_active": 5,
    "hyd_common": 10,
    "slew_active": 5,
    "slew_power": 60,
    "thermal_drive": 30,
    "thermal_motor": 300,
    "thermal_resistor": 120,
}

activity_thresholds = {
    "always_on": 0.2,
    "boom_active": 0.3,
    "hoist_active": 0.3,
    "hyd_common": 0.3,
    "slew_active": 0.3,
    "slew_power": 0.3,
    "thermal_drive": 0.2,
    "thermal_motor": 0.2,
    "thermal_resistor": 0.2,
}

# ============================================================
# HELPERS
# ============================================================

def parse_lift_group_bucket(index_val):
    parts = str(index_val).split("_")
    lift_id = parts[0]
    bucket_num = parts[-1]
    return lift_id, int(bucket_num)

def robust_lower_threshold(values, method="iqr", factor=3.0):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    if method == "iqr":
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        return q1 - factor * iqr

    if method == "mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            return np.min(x) - 1e-9
        return med - factor * 1.4826 * mad

    raise ValueError("method must be 'iqr' or 'mad'")

def compute_row_activity(pdf, threshold=0.3):
    if pdf.shape[1] == 0:
        return pd.Series(False, index=pdf.index)

    row_max = np.abs(pdf.values).max(axis=1)
    return pd.Series(row_max > threshold, index=pdf.index)

# ============================================================
# STEP 1: PREPARE ALL GROUP DATAFRAMES
# Uses df_null_dict from preprocessing
# ============================================================

group_pdfs = {}
group_scalers = {}
group_activity = {}
group_feature_names = {}
group_row_counts = {}

if "df_null_dict" not in globals():
    raise NameError("df_null_dict does not exist. Run the NULL preprocessing script first.")

for group, pdf in df_null_dict.items():
    print(f"\nPreparing group: {group}")
    pdf = pdf.copy()

    if "bucket_start_unix" not in pdf.columns:
        raise ValueError(f"Group '{group}' must contain 'bucket_start_unix'")

    if pdf.index.name != "lift_group_bucket":
        if "lift_group_bucket" in pdf.columns:
            pdf = pdf.set_index("lift_group_bucket")
        else:
            raise ValueError(f"Group '{group}' must have index or column 'lift_group_bucket'")

    pdf = pdf.sort_values("bucket_start_unix")

    feature_df = pdf.drop(columns=["bucket_start_unix"], errors="ignore").copy()
    feature_df.columns = [c.replace("ImportS7Device.", "") for c in feature_df.columns]
    feature_df = feature_df.select_dtypes(include=[np.number])

    if feature_df.shape[1] == 0:
        print(f"  Skipping group '{group}' because it has no numeric columns.")
        continue

    # NULL experiment: no filling. Drop rows with any NULL sensor value.
    before_rows = len(feature_df)
    feature_df = feature_df.dropna(axis=0, how="any")
    after_rows = len(feature_df)

    print(f"  Dropped NULL rows: {before_rows - after_rows} / {before_rows}")
    print(f"  Remaining complete rows: {after_rows}")

    if feature_df.shape[0] < 10:
        print(f"  Skipping group '{group}' because too few complete rows remain.")
        continue

    scaler = StandardScaler()
    feature_scaled = pd.DataFrame(
        scaler.fit_transform(feature_df),
        index=feature_df.index,
        columns=feature_df.columns
    )

    activity_mask = compute_row_activity(
        feature_scaled,
        threshold=activity_thresholds.get(group, 0.3)
    )

    group_pdfs[group] = feature_scaled
    group_scalers[group] = scaler
    group_activity[group] = activity_mask
    group_feature_names[group] = list(feature_scaled.columns)
    group_row_counts[group] = len(feature_scaled)

    print(f"  Final shape: {feature_scaled.shape}")
    print(f"  Active rows: {activity_mask.sum()} / {len(activity_mask)}")

if len(group_pdfs) == 0:
    raise ValueError("No groups have enough complete NULL-free rows for ML.")

# ============================================================
# STEP 2: TRAIN ONE SPECIALIST PER GROUP
# ============================================================

specialists = {}
specialist_scores = {}
specialist_thresholds = {}

for group, pdf in group_pdfs.items():
    print(f"\nTraining specialist: {group}")

    activity_mask = group_activity[group]
    train_df = pdf.loc[activity_mask].copy()

    if len(train_df) < 50:
        print(f"  Too few active rows ({len(train_df)}), training on all rows instead.")
        train_df = pdf.copy()

    if len(train_df) < 10:
        print(f"  Skipping specialist '{group}' because too few training rows.")
        continue

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(train_df.values)

    raw_scores = model.decision_function(pdf.values)

    active_scores = raw_scores[activity_mask.values] if activity_mask.sum() > 0 else raw_scores
    threshold = robust_lower_threshold(active_scores, method="iqr", factor=3.0)

    if not np.isfinite(threshold):
        threshold = np.min(raw_scores) - 1e-9

    pred = np.where((activity_mask.values) & (raw_scores < threshold), -1, 1)

    specialists[group] = model
    specialist_thresholds[group] = float(threshold)

    specialist_scores[group] = pd.DataFrame({
        f"{group}_score": raw_scores,
        f"{group}_pred": pred,
        f"{group}_active": activity_mask.astype(int).values,
        "bucket_start_unix": [parse_lift_group_bucket(idx)[1] for idx in pdf.index]
    }, index=pdf.index)

    n_anomalies = (pred == -1).sum()
    print(f"  Threshold: {threshold:.5f}")
    print(f"  Anomalies found: {n_anomalies} / {len(pred)}")

if len(specialist_scores) == 0:
    raise ValueError("No specialist models were trained.")

# ============================================================
# STEP 3: BUILD PER-LIFT TIMESTAMPED SPECIALIST OUTPUTS
# ============================================================

specialist_with_lift = {}

for group, scores_df in specialist_scores.items():
    temp = scores_df.copy()

    parsed = [parse_lift_group_bucket(idx) for idx in temp.index]
    temp["lift_id"] = [p[0] for p in parsed]
    temp["bucket_unix"] = [p[1] for p in parsed]
    temp["timestamp"] = pd.to_datetime(temp["bucket_unix"], unit="s", utc=True)

    specialist_with_lift[group] = (
        temp.reset_index()
        .rename(columns={"index": "lift_group_bucket"})
        .sort_values(["lift_id", "timestamp"])
        .reset_index(drop=True)
    )

# ============================================================
# STEP 4: ALIGN ALL SPECIALISTS ONTO A COMMON BASE TIMELINE
# Pick fastest available group, not necessarily fastest configured group
# ============================================================

available_groups = list(specialist_with_lift.keys())
base_group = min(available_groups, key=lambda g: bucket_sizes.get(g, 999999))

print(f"\nBase timeline group: {base_group} ({bucket_sizes.get(base_group)}s buckets)")

meta_features = specialist_with_lift[base_group][[
    "lift_group_bucket", "lift_id", "timestamp",
    f"{base_group}_score", f"{base_group}_pred", f"{base_group}_active"
]].copy()

meta_features = meta_features.sort_values(["lift_id", "timestamp"]).reset_index(drop=True)

for group, scores_df in specialist_with_lift.items():
    if group == base_group:
        continue

    print(f"Merging group: {group}")

    right = scores_df[[
        "lift_id", "timestamp",
        f"{group}_score", f"{group}_pred", f"{group}_active"
    ]].sort_values(["lift_id", "timestamp"]).reset_index(drop=True)

    merged_parts = []

    for lid in meta_features["lift_id"].unique():
        left_lift = meta_features[meta_features["lift_id"] == lid].copy()
        right_lift = right[right["lift_id"] == lid].copy()

        if len(right_lift) == 0:
            left_lift[f"{group}_score"] = np.nan
            left_lift[f"{group}_pred"] = np.nan
            left_lift[f"{group}_active"] = np.nan
            merged_parts.append(left_lift)
            continue

        merged = pd.merge_asof(
            left_lift.sort_values("timestamp"),
            right_lift.sort_values("timestamp"),
            on="timestamp",
            direction="backward"
        )

        if "lift_id_x" in merged.columns:
            merged["lift_id"] = merged["lift_id_x"]
            merged = merged.drop(
                columns=[c for c in ["lift_id_x", "lift_id_y"] if c in merged.columns]
            )

        merged_parts.append(merged)

    meta_features = pd.concat(merged_parts, ignore_index=True)
    print(f"  Meta feature shape after merge: {meta_features.shape}")

active_cols = [c for c in meta_features.columns if c.endswith("_active")]
meta_features["n_active_groups"] = meta_features[active_cols].fillna(0).sum(axis=1)

meta_features = meta_features[meta_features["n_active_groups"] > 0].copy()
meta_features = meta_features.reset_index(drop=True)

print(f"\nFinal meta feature matrix before meta ML: {meta_features.shape}")
print(f"Unique lifts: {meta_features['lift_id'].nunique() if len(meta_features) else 0}")

if len(meta_features) == 0:
    raise ValueError("No active meta rows remain after NULL-row filtering.")

print(meta_features.head(10))

# ============================================================
# STEP 5: TRAIN META MODEL ON STACKED SPECIALIST OUTPUTS
# ============================================================

score_columns = [c for c in meta_features.columns if c.endswith("_score")]
pred_columns = [c for c in meta_features.columns if c.endswith("_pred")]
meta_input_cols = score_columns + pred_columns

X_meta_df = meta_features[meta_input_cols].copy()

# After time alignment, some specialist outputs are missing.
# This is not raw sensor filling; this only fills missing specialist scores.
X_meta_df = X_meta_df.fillna(X_meta_df.median(numeric_only=True))
X_meta_df = X_meta_df.fillna(0)

if len(X_meta_df) < 10:
    raise ValueError("Too few meta rows to train meta model.")

X_meta = X_meta_df.values

meta_model = IsolationForest(
    n_estimators=400,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

meta_model.fit(X_meta)

meta_scores = meta_model.decision_function(X_meta)

active_meta_mask = meta_features["n_active_groups"] >= 2
if active_meta_mask.sum() < 50:
    active_meta_mask = meta_features["n_active_groups"] >= 1

meta_threshold = robust_lower_threshold(
    meta_scores[active_meta_mask.values],
    method="iqr",
    factor=3.0
)

if not np.isfinite(meta_threshold):
    meta_threshold = np.min(meta_scores) - 1e-9

meta_features["meta_score"] = meta_scores
meta_features["meta_pred"] = np.where(
    (meta_features["n_active_groups"] > 0) &
    (meta_features["meta_score"] < meta_threshold),
    -1,
    1
)

n_meta_anomalies = (meta_features["meta_pred"] == -1).sum()

print(f"\nMeta threshold: {meta_threshold:.5f}")
print(f"Meta model anomalies: {n_meta_anomalies} / {len(meta_features)}")

# ============================================================
# STEP 6: RESULTS PER LIFT
# ============================================================

print("\n" + "=" * 60)
print("ANOMALIES PER LIFT")
print("=" * 60)

anomaly_summary = (
    meta_features
    .groupby("lift_id")
    .agg(
        total_samples=("meta_pred", "count"),
        anomalies=("meta_pred", lambda x: (x == -1).sum()),
        worst_score=("meta_score", "min"),
        mean_score=("meta_score", "mean"),
        active_groups_mean=("n_active_groups", "mean")
    )
)

anomaly_summary["anomaly_pct"] = (
    100 * anomaly_summary["anomalies"] / anomaly_summary["total_samples"]
)

anomaly_summary = anomaly_summary.sort_values(
    ["anomalies", "worst_score"],
    ascending=[False, True]
)

print(anomaly_summary.to_string())

# ============================================================
# STEP 7: JOIN MOVEMENT TYPE
# ============================================================

if "df_liftlog" in globals():
    liftlog_pd = df_liftlog.select("lift_id", "movementtype").distinct().toPandas()
    liftid_to_movementtype = dict(
        zip(liftlog_pd["lift_id"].astype(str), liftlog_pd["movementtype"])
    )

    anomaly_summary = anomaly_summary.copy()
    anomaly_summary.index = anomaly_summary.index.astype(str)
    anomaly_summary["movementtype"] = anomaly_summary.index.map(liftid_to_movementtype.get)

    print("\n" + "=" * 60)
    print("ANOMALY SUMMARY WITH MOVEMENT TYPE")
    print("=" * 60)
    print(anomaly_summary.to_string())

    from pyspark.sql.functions import col

    anomaly_liftids = list(anomaly_summary.index)

    df_anomaly_lifts = (
        df_liftlog
        .select("lift_id", "movementtype")
        .distinct()
        .filter(col("lift_id").cast("string").isin(anomaly_liftids))
    )

    display(df_anomaly_lifts)
else:
    print("\ndf_liftlog not found. Skipping movementtype join.")

# ============================================================
# STEP 8: VISUALIZE WORST LIFT
# ============================================================

if len(anomaly_summary) > 0:
    worst_lift = anomaly_summary.index[0]

    lift_data = meta_features[
        meta_features["lift_id"].astype(str) == str(worst_lift)
    ].copy()

    lift_data = lift_data.sort_values("timestamp")
    anomaly_mask = lift_data["meta_pred"] == -1

    n_plots = len(score_columns) + 1

    fig, axes = plt.subplots(
        n_plots,
        1,
        figsize=(20, 3 * n_plots),
        sharex=True
    )

    if n_plots == 1:
        axes = [axes]

    for i, col_name in enumerate(score_columns):
        ax = axes[i]
        ax.plot(
            lift_data["timestamp"],
            lift_data[col_name],
            linewidth=0.7,
            alpha=0.8
        )
        ax.scatter(
            lift_data.loc[anomaly_mask, "timestamp"],
            lift_data.loc[anomaly_mask, col_name],
            c="red",
            s=10,
            alpha=0.7
        )
        ax.set_ylabel(col_name.replace("_score", ""), fontsize=8)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    ax = axes[-1]
    ax.plot(
        lift_data["timestamp"],
        lift_data["meta_score"],
        linewidth=0.9,
        color="black"
    )
    ax.scatter(
        lift_data.loc[anomaly_mask, "timestamp"],
        lift_data.loc[anomaly_mask, "meta_score"],
        c="red",
        s=10,
        alpha=0.7
    )
    ax.axhline(y=meta_threshold, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("META", fontsize=10, fontweight="bold")

    plt.suptitle(f"LIFT {worst_lift} - NULL-row Stacked Anomaly Detection", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print(f"WHAT DRIVES ANOMALIES ON LIFT {worst_lift}?")
    print("=" * 60)

    for col_name in score_columns:
        normal_mean = lift_data.loc[~anomaly_mask, col_name].mean()
        anomaly_mean = lift_data.loc[anomaly_mask, col_name].mean()
        print(
            f"{col_name:40s}  normal={normal_mean:.3f}  anomaly={anomaly_mean:.3f}"
        )
else:
    print("No lifts available for visualization.")

# ============================================================
# STEP 9: SAVE ALL MODELS
# ============================================================

crane_files_dir = "/Volumes/craneds_dev/maxedge_lhdp/crane_files"
MODEL_DIR = os.path.join(crane_files_dir, "anomaly_model_null_rows")
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\nSaving all models to: {MODEL_DIR}")
print("=" * 60)

config = {
    "experiment": "null_rows_no_sensor_fill",
    "bucket_sizes": bucket_sizes,
    "activity_thresholds": activity_thresholds,
    "specialist_thresholds": specialist_thresholds,
    "meta_threshold": float(meta_threshold),
    "base_group": base_group,
    "score_columns": score_columns,
    "pred_columns": pred_columns,
    "group_feature_names": group_feature_names,
    "group_row_counts_after_null_drop": group_row_counts,
}

with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("  saved config.json")

for group, scaler in group_scalers.items():
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{group}.joblib"))
    print(f"  saved scaler_{group}.joblib")

for group, model in specialists.items():
    joblib.dump(model, os.path.join(MODEL_DIR, f"specialist_{group}.joblib"))
    print(f"  saved specialist_{group}.joblib")

joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.joblib"))
print("  saved meta_model.joblib")

print("\n" + "=" * 60)
print("SAVED FILES:")
print("=" * 60)

total_size = 0

for f_name in sorted(os.listdir(MODEL_DIR)):
    f_path = os.path.join(MODEL_DIR, f_name)
    size_kb = os.path.getsize(f_path) / 1024
    total_size += size_kb
    print(f"  {f_name:45s} {size_kb:>10.1f} KB")

print(f"  {'TOTAL':45s} {total_size:>10.1f} KB")
print(f"\nPath: {MODEL_DIR}")

# COMMAND ----------

# Check if any lift is classified as an anomaly
anomalous_lifts = anomaly_summary[anomaly_summary["anomalies"] > 0]
if not anomalous_lifts.empty:
    print("Lifts classified as anomalies:")
    print(anomalous_lifts.index.tolist())
else:
    print("No lifts classified as anomalies.")