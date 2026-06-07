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
# 12) RULE-BASED ZERO FILL + DISTANCE-WEIGHTED INTERPOLATION
# ============================================================

def weighted_fill_series(values, max_gap=10, zero_gap_fill=True):
    """
    Fill NaNs in a 1D numpy array using rule-based logic:

    1) If a consecutive NaN gap is bounded by 0 on both sides and
       zero_gap_fill=True, fill that gap with 0.
    2) Else, if the gap length <= max_gap, fill using inverse-distance
       weighted interpolation.
    3) Else leave as NaN.

    Parameters
    ----------
    values : np.ndarray
        1D numeric array with NaNs
    max_gap : int
        Maximum consecutive gap length to interpolate
    zero_gap_fill : bool
        Whether zero-bounded gaps should be filled with zero

    Returns
    -------
    np.ndarray
        Filled array
    """
    filled = values.astype(float).copy()
    n = len(filled)

    i = 0
    while i < n:
        if not np.isnan(filled[i]):
            i += 1
            continue

        # Find consecutive missing gap
        start = i
        while i < n and np.isnan(filled[i]):
            i += 1
        end = i - 1

        left_idx = start - 1
        right_idx = end + 1

        left_valid = left_idx >= 0 and not np.isnan(filled[left_idx])
        right_valid = right_idx < n and not np.isnan(filled[right_idx])

        gap_length = end - start + 1

        # Case 1: inactive zero-period
        if (
            zero_gap_fill
            and left_valid
            and right_valid
            and filled[left_idx] == 0
            and filled[right_idx] == 0
        ):
            filled[start:end + 1] = 0
            continue

        # Case 2: short-gap interpolation
        if gap_length <= max_gap:
            for k in range(start, end + 1):
                neighbors = []
                weights = []

                if left_valid:
                    d_left = k - left_idx
                    neighbors.append(filled[left_idx])
                    weights.append(1.0 / d_left)

                if right_valid:
                    d_right = right_idx - k
                    neighbors.append(filled[right_idx])
                    weights.append(1.0 / d_right)

                if weights:
                    filled[k] = np.dot(neighbors, weights) / np.sum(weights)

    return filled


def extract_lift_id(index_series):
    """
    Extract lift_id from index like:
        12345_1710000000
    """
    extracted = index_series.astype(str).str.extract(r"^(\d+)_")[0]
    return extracted

# ============================================================
# 12A) GROUP-SPECIFIC INTERPOLATION WINDOWS
# ============================================================

fill_windows = {
    "always_on": 6,
    "boom_active": 6,
    "hoist_active": 6,
    "hyd_common": 4,
    "slew_active": 6,
    "slew_power": 2,
    "thermal_drive": 3,
    "thermal_motor": 2,
    "thermal_resistor": 2,
}

# ============================================================
# 12B) COLUMNS WHERE ZERO = INACTIVE / UNLOADED STATE
# ============================================================

zero_fill_columns = {
    # always_on
    "ImportS7Device.CBM.LuffingWinch_Moment",
    "ImportS7Device.CBM.MainHoistWinch_Moment",
    "ImportS7Device.CBM.SlewBearing_Moment",
    "ImportS7Device.General.BoomLoad",
    "ImportS7Device.General.HoistLoadPctSWL",
    "ImportS7Device.General.PedestalMoment",
    "ImportS7Device.SafetySystems.AOPS RopeForce",

    # boom_active
    "ImportS7Device.Drives.Boom_Status.ActualSpeed",
    "ImportS7Device.Drives.Boom_Status.ActualPower",
    "ImportS7Device.Drives.Boom_Status.ActualCurrent",
    "ImportS7Device.Drives.Boom_Status.ActualTorque",
    "ImportS7Device.Drives.Boom_Control.SpeedReference",
    "ImportS7Device.Joysticks.Boom",

    # hoist_active
    "ImportS7Device.Drives.Hoist_Status.ActualSpeed",
    "ImportS7Device.Drives.Hoist_Status.ActualPower",
    "ImportS7Device.Drives.Hoist_Status.ActualCurrent",
    "ImportS7Device.Drives.Hoist_Control.SpeedReference",
    "ImportS7Device.Joysticks.Hoist",

    # hyd_common
    "ImportS7Device.Drives.Hoist_Status.ActualTorque",

    # slew_active
    "ImportS7Device.Drives.SlewA_Status.ActualSpeed",
    "ImportS7Device.Drives.SlewA_Status.ActualTorque",
    "ImportS7Device.Drives.SlewA_Status.DroopFeedbackSpeedReduction",
    "ImportS7Device.Drives.SlewB_Status.ActualSpeed",
    "ImportS7Device.Drives.SlewB_Status.ActualTorque",
    "ImportS7Device.Drives.SlewB_Status.ActualCurrent",
    "ImportS7Device.Joysticks.Slew",
    "ImportS7Device.General.SlewMoment",
    "ImportS7Device.CBM.SlewGear_Moment",
    "ImportS7Device.Drives.SlewA_Status.ActualCurrent",
    "ImportS7Device.Drives.SlewA_Control.SpeedReference",
    "ImportS7Device.Drives.SlewB_Status.DroopFeedbackSpeedReduction",
    "ImportS7Device.Drives.SlewB_Control.SpeedReference",

    # slew_power
    "ImportS7Device.Drives.SlewA_Status.ActualPower",
    "ImportS7Device.Drives.SlewB_Status.ActualPower",
    "ImportS7Device.Drives.SlewA_Status.TorqueLimEffective",
    "ImportS7Device.Drives.SlewB_Status.TorqueLimEffective",
}

# ============================================================
# 12C) APPLY FILLING GROUP-BY-GROUP, LIFT-BY-LIFT
# ============================================================

df_filled_dict = {}

for group, spark_df in df_pivoted_dict.items():
    print(f"\nFilling group: {group}")

    pdf = spark_df.toPandas()

    pdf = pdf.sort_values(["lift_group_bucket", "bucket_start_unix"]).copy()
    pdf = pdf.set_index("lift_group_bucket", drop=True)

    pdf["_lift_id"] = extract_lift_id(pdf.index.to_series())

    if pdf["_lift_id"].isna().any():
        raise ValueError(f"Could not extract lift_id for all rows in group '{group}'")

    data_cols = [c for c in pdf.columns if c not in ["bucket_start_unix", "_lift_id"]]
    max_gap = fill_windows.get(group, 5)

    filled_parts = []

    for lift_id, subdf in pdf.groupby("_lift_id", sort=False):
        subdf = subdf.sort_values("bucket_start_unix").copy()

        for col in data_cols:
            use_zero_gap_fill = col in zero_fill_columns
            subdf[col] = weighted_fill_series(
                subdf[col].values,
                max_gap=max_gap,
                zero_gap_fill=use_zero_gap_fill
            )

        filled_parts.append(subdf)

    pdf_filled = pd.concat(filled_parts).sort_values(["_lift_id", "bucket_start_unix"])
    pdf_filled = pdf_filled.drop(columns=["_lift_id"])

    df_filled_dict[group] = pdf_filled

# ============================================================
# 13) OPTIONAL: CHECK RESULT
# ============================================================

for group, pdf in df_filled_dict.items():
    print(f"\n--- {group} ---")
    print(pdf.head())
    break

# Example access:
# display(spark.createDataFrame(df_filled_dict["always_on"].reset_index()))

# COMMAND ----------

display(spark.createDataFrame(df_filled_dict["always_on"].reset_index()))

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

# ------------------------------------------------------------
# Activity thresholds per group
# These are used to avoid detecting anomalies during inactive periods.
# Start simple: row is active if any scaled absolute feature exceeds threshold.
# ------------------------------------------------------------
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
    """
    Parse index like:
        liftid_bucketunix
    """
    parts = str(index_val).split("_")
    lift_id = parts[0]
    bucket_num = parts[-1]
    return lift_id, int(bucket_num)

def robust_lower_threshold(values, method="iqr", factor=3.0):
    """
    Compute a robust lower-tail threshold for anomaly scores.
    Lower scores = more anomalous.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    if method == "iqr":
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        return q1 - factor * iqr

    elif method == "mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            return np.min(x) - 1e-9
        return med - factor * 1.4826 * mad

    else:
        raise ValueError("method must be 'iqr' or 'mad'")

def compute_row_activity(pdf, threshold=0.3):
    """
    Determine whether each row has meaningful activity.
    Uses the max absolute standardized feature value in the row.
    """
    if pdf.shape[1] == 0:
        return pd.Series(False, index=pdf.index)

    row_max = np.abs(pdf.values).max(axis=1)
    return pd.Series(row_max > threshold, index=pdf.index)

# ============================================================
# STEP 1: PREPARE ALL GROUP DATAFRAMES
# Uses df_filled_dict from preprocessing
# ============================================================

group_pdfs = {}
group_scalers = {}
group_activity = {}
group_feature_names = {}

for group, pdf in df_filled_dict.items():
    print(f"Preparing group: {group}")
    pdf = pdf.copy()

    if "bucket_start_unix" not in pdf.columns:
        raise ValueError(f"Group '{group}' must contain 'bucket_start_unix'")

    if pdf.index.name != "lift_group_bucket":
        if "lift_group_bucket" in pdf.columns:
            pdf = pdf.set_index("lift_group_bucket")
        else:
            raise ValueError(f"Group '{group}' must have index or column 'lift_group_bucket'")

    pdf = pdf.sort_values("bucket_start_unix")

    # Keep bucket_start_unix separately for metadata
    bucket_start_unix = pdf["bucket_start_unix"].copy()

    # Keep only sensor columns
    feature_df = pdf.drop(columns=["bucket_start_unix"], errors="ignore").copy()

    # Clean column names
    feature_df.columns = [c.replace("ImportS7Device.", "") for c in feature_df.columns]

    # Keep only numeric columns
    feature_df = feature_df.select_dtypes(include=[np.number])

    if feature_df.shape[1] == 0:
        print(f"  Skipping group '{group}' because it has no numeric columns.")
        continue

    # Final fallback fill for ML
    feature_df = feature_df.fillna(feature_df.median())

    # Scale
    scaler = StandardScaler()
    feature_scaled = pd.DataFrame(
        scaler.fit_transform(feature_df),
        index=feature_df.index,
        columns=feature_df.columns
    )

    # Activity mask
    activity_mask = compute_row_activity(
        feature_scaled,
        threshold=activity_thresholds.get(group, 0.3)
    )

    # Store
    group_pdfs[group] = feature_scaled
    group_scalers[group] = scaler
    group_activity[group] = activity_mask
    group_feature_names[group] = list(feature_scaled.columns)

    print(f"  Final shape: {feature_scaled.shape}")
    print(f"  Active rows: {activity_mask.sum()} / {len(activity_mask)}")

# ============================================================
# STEP 2: TRAIN ONE SPECIALIST PER GROUP
# No fixed anomaly percentage
# ============================================================

specialists = {}
specialist_scores = {}
specialist_thresholds = {}

for group, pdf in group_pdfs.items():
    print(f"\nTraining specialist: {group}")

    activity_mask = group_activity[group]

    # Train only on rows with activity if possible
    train_df = pdf.loc[activity_mask].copy()

    # Fallback: if too few active rows, train on all rows
    if len(train_df) < 50:
        print(f"  Too few active rows ({len(train_df)}), training on all rows instead.")
        train_df = pdf.copy()

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(train_df.values)

    # Score all rows
    raw_scores = model.decision_function(pdf.values)

    # Robust threshold from active rows only
    active_scores = raw_scores[activity_mask.values] if activity_mask.sum() > 0 else raw_scores
    threshold = robust_lower_threshold(active_scores, method="iqr", factor=3.0)

    # Anomaly only if active and below threshold
    pred = np.where((activity_mask.values) & (raw_scores < threshold), -1, 1)

    specialists[group] = model
    specialist_thresholds[group] = threshold

    specialist_scores[group] = pd.DataFrame({
        f"{group}_score": raw_scores,
        f"{group}_pred": pred,
        f"{group}_active": activity_mask.astype(int).values,
        "bucket_start_unix": [
            parse_lift_group_bucket(idx)[1] for idx in pdf.index
        ]
    }, index=pdf.index)

    n_anomalies = (pred == -1).sum()
    print(f"  Threshold: {threshold:.5f}")
    print(f"  Anomalies found: {n_anomalies} / {len(pred)}")

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

    specialist_with_lift[group] = temp.reset_index().rename(columns={"index": "lift_group_bucket"})

# ============================================================
# STEP 4: ALIGN ALL SPECIALISTS ONTO A COMMON BASE TIMELINE
# ============================================================

base_group = min(bucket_sizes, key=bucket_sizes.get)
print(f"\nBase timeline group: {base_group} ({bucket_sizes[base_group]}s buckets)")

if base_group not in specialist_with_lift:
    raise ValueError(f"Base group '{base_group}' not found in specialist outputs.")

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

        # merge_asof may duplicate / overwrite lift_id columns
        if "lift_id_x" in merged.columns:
            merged["lift_id"] = merged["lift_id_x"]
            merged = merged.drop(columns=[c for c in ["lift_id_x", "lift_id_y"] if c in merged.columns])

        merged_parts.append(merged)

    meta_features = pd.concat(merged_parts, ignore_index=True)
    print(f"  Meta feature shape after merge: {meta_features.shape}")

# Keep rows where at least one specialist is active
active_cols = [c for c in meta_features.columns if c.endswith("_active")]
meta_features["n_active_groups"] = meta_features[active_cols].fillna(0).sum(axis=1)

meta_features = meta_features[meta_features["n_active_groups"] > 0].copy()
meta_features = meta_features.reset_index(drop=True)

print(f"\nFinal meta feature matrix: {meta_features.shape}")
print(f"Unique lifts: {meta_features['lift_id'].nunique()}")
print(meta_features.head(10))

# ============================================================
# STEP 5: TRAIN META MODEL ON STACKED SPECIALIST OUTPUTS
# No fixed contamination percentage
# ============================================================

score_columns = [c for c in meta_features.columns if c.endswith("_score")]
pred_columns = [c for c in meta_features.columns if c.endswith("_pred")]
meta_input_cols = score_columns + pred_columns

# Fill missing specialist outputs after alignment
X_meta_df = meta_features[meta_input_cols].copy()
X_meta_df = X_meta_df.fillna(X_meta_df.median())

X_meta = X_meta_df.values

meta_model = IsolationForest(
    n_estimators=400,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

meta_model.fit(X_meta)

meta_scores = meta_model.decision_function(X_meta)

# Build robust threshold from rows with at least 2 active specialists if possible
active_meta_mask = meta_features["n_active_groups"] >= 2
if active_meta_mask.sum() < 50:
    active_meta_mask = meta_features["n_active_groups"] >= 1

meta_threshold = robust_lower_threshold(
    meta_scores[active_meta_mask.values],
    method="iqr",
    factor=3.0
)

meta_features["meta_score"] = meta_scores
meta_features["meta_pred"] = np.where(
    (meta_features["n_active_groups"] > 0) & (meta_features["meta_score"] < meta_threshold),
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

liftlog_pd = df_liftlog.select("lift_id", "movementtype").distinct().toPandas()
liftid_to_movementtype = dict(zip(liftlog_pd["lift_id"].astype(str), liftlog_pd["movementtype"]))

anomaly_summary = anomaly_summary.copy()
anomaly_summary.index = anomaly_summary.index.astype(str)
anomaly_summary["movementtype"] = anomaly_summary.index.map(liftid_to_movementtype.get)

print("\n" + "=" * 60)
print("ANOMALY SUMMARY WITH MOVEMENT TYPE")
print("=" * 60)
print(anomaly_summary.to_string())

# Optional spark display
from pyspark.sql.functions import col

anomaly_liftids = list(anomaly_summary.index)
df_anomaly_lifts = (
    df_liftlog
    .select("lift_id", "movementtype")
    .distinct()
    .filter(col("lift_id").cast("string").isin(anomaly_liftids))
)

display(df_anomaly_lifts)

# ============================================================
# STEP 8: VISUALIZE WORST LIFT
# ============================================================

if len(anomaly_summary) > 0:
    worst_lift = anomaly_summary.index[0]
    lift_data = meta_features[meta_features["lift_id"].astype(str) == str(worst_lift)].copy()
    lift_data = lift_data.sort_values("timestamp")

    anomaly_mask = lift_data["meta_pred"] == -1

    fig, axes = plt.subplots(
        len(score_columns) + 1,
        1,
        figsize=(20, 3 * (len(score_columns) + 1)),
        sharex=True
    )

    if len(score_columns) == 1:
        axes = [axes]

    for i, col in enumerate(score_columns):
        ax = axes[i]
        ax.plot(lift_data["timestamp"], lift_data[col], linewidth=0.7, alpha=0.8)
        ax.scatter(
            lift_data.loc[anomaly_mask, "timestamp"],
            lift_data.loc[anomaly_mask, col],
            c="red",
            s=10,
            alpha=0.7
        )
        ax.set_ylabel(col.replace("_score", ""), fontsize=8)
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

    plt.suptitle(f"LIFT {worst_lift} - Stacked Anomaly Detection", fontsize=14)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Which subsystem drives anomalies?
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"WHAT DRIVES ANOMALIES ON LIFT {worst_lift}?")
    print("=" * 60)

    for col in score_columns:
        normal_mean = lift_data.loc[~anomaly_mask, col].mean()
        anomaly_mean = lift_data.loc[anomaly_mask, col].mean()
        print(f"{col:40s}  normal={normal_mean:.3f}  anomaly={anomaly_mean:.3f}")
else:
    print("No lifts available for visualization.")

# ============================================================
# STEP 9: SAVE ALL MODELS
# ============================================================

crane_files_dir = "/Volumes/craneds_dev/maxedge_lhdp/crane_files"
MODEL_DIR = os.path.join(crane_files_dir, "anomaly_model")
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\nSaving all models to: {MODEL_DIR}")
print("=" * 60)

config = {
    "bucket_sizes": bucket_sizes,
    "activity_thresholds": activity_thresholds,
    "specialist_thresholds": specialist_thresholds,
    "meta_threshold": float(meta_threshold),
    "base_group": base_group,
    "score_columns": score_columns,
    "pred_columns": pred_columns,
    "group_feature_names": group_feature_names
}

with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print("  ✓ config.json")

for group, scaler in group_scalers.items():
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{group}.joblib"))
    print(f"  ✓ scaler_{group}.joblib")

for group, model in specialists.items():
    joblib.dump(model, os.path.join(MODEL_DIR, f"specialist_{group}.joblib"))
    print(f"  ✓ specialist_{group}.joblib")

joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.joblib"))
print("  ✓ meta_model.joblib")

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

lift_id_query = "8961"

# Check if lift_id 10246 is in the anomaly summary
if lift_id_query in anomaly_summary.index:
    print(f"Anomaly summary for lift_id {lift_id_query}:")
    print(anomaly_summary.loc[lift_id_query])
else:
    print(f"lift_id {lift_id_query} not found in anomaly summary.")

# Show all meta_features rows for lift_id 10246
meta_10246 = meta_features[meta_features["lift_id"].astype(str) == lift_id_query]
print(f"\nMeta features for lift_id {lift_id_query}:")
print(meta_10246.head())

# Optionally display in Databricks
if not meta_10246.empty:
    display(spark.createDataFrame(meta_10246))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col

# Use the alarm table from the main context, not the overload fit table
df_alarms = df_alarms_with_lift  # from Cell 2

most_anom_lift = str(anomaly_summary.index[0])  # anomaly_summary from Cell 5

df_alarms_most_anom = (
    df_alarms
    .filter(col("lift_id").cast("string") == F.lit(most_anom_lift))
)

print("Most anomalous lift:", most_anom_lift)
print("Alarm rows for this lift:", df_alarms_most_anom.count())

display(
    df_alarms_most_anom
    .groupBy("id", "text")
    .agg(
        F.count("*").alias("alarm_count"),
        F.min("timestamp_utc").alias("first_alarm"),
        F.max("timestamp_utc").alias("last_alarm")
    )
    .orderBy(F.desc("alarm_count"))
)