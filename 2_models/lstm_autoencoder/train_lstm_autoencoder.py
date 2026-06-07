# Databricks notebook source
# MAGIC %md
# MAGIC Here are the databases load so computation is already done with refering raw data with an lift id

# COMMAND ----------

# MAGIC %pip install tensorflow
# MAGIC dbutils.library.restartPython()

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

# Databricks note:
# If tensorflow is missing, run once in a notebook cell:
# %pip install tensorflow
# dbutils.library.restartPython()

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping


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

sequence_lengths = {
    "always_on": 24,
    "boom_active": 24,
    "hoist_active": 24,
    "hyd_common": 12,
    "slew_active": 24,
    "slew_power": 8,
    "thermal_drive": 8,
    "thermal_motor": 6,
    "thermal_resistor": 8,
}

EPOCHS = 40
BATCH_SIZE = 128
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# HELPERS
# ============================================================

def parse_lift_group_bucket(index_val):
    parts = str(index_val).split("_")
    return parts[0], int(parts[-1])


def robust_upper_threshold(values, method="iqr", factor=3.0):
    """
    For autoencoders:
    Higher reconstruction error = more anomalous.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    if method == "iqr":
        q1 = np.percentile(x, 25)
        q3 = np.percentile(x, 75)
        iqr = q3 - q1
        return q3 + factor * iqr

    if method == "mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            return np.max(x) + 1e-9
        return med + factor * 1.4826 * mad

    raise ValueError("method must be 'iqr' or 'mad'")


def compute_row_activity(pdf, threshold=0.3):
    if pdf.shape[1] == 0:
        return pd.Series(False, index=pdf.index)

    row_max = np.abs(pdf.values).max(axis=1)
    return pd.Series(row_max > threshold, index=pdf.index)


def make_lstm_autoencoder(seq_len, n_features, latent_dim=None):
    if latent_dim is None:
        latent_dim = max(4, min(64, n_features * 2))

    inp = Input(shape=(seq_len, n_features))

    x = LSTM(latent_dim, activation="tanh", return_sequences=False)(inp)
    x = RepeatVector(seq_len)(x)
    x = LSTM(latent_dim, activation="tanh", return_sequences=True)(x)
    out = TimeDistributed(Dense(n_features))(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")

    return model


def build_sequences_per_lift(feature_df, activity_mask, seq_len):
    """
    Builds rolling sequences per lift.
    Each sequence is assigned to the timestamp/index of its last row.
    """
    X_seq = []
    end_indices = []
    seq_active = []

    temp = feature_df.copy()
    temp["_lift_id"] = [
        parse_lift_group_bucket(idx)[0] for idx in temp.index
    ]
    temp["_bucket_unix"] = [
        parse_lift_group_bucket(idx)[1] for idx in temp.index
    ]
    temp["_active"] = activity_mask.astype(bool).values

    for lift_id, subdf in temp.groupby("_lift_id", sort=False):
        subdf = subdf.sort_values("_bucket_unix")

        feature_cols = [
            c for c in subdf.columns
            if c not in ["_lift_id", "_bucket_unix", "_active"]
        ]

        values = subdf[feature_cols].values
        active_values = subdf["_active"].values
        idx_values = subdf.index.to_numpy()

        if len(subdf) < seq_len:
            continue

        for i in range(seq_len - 1, len(subdf)):
            start = i - seq_len + 1
            end = i + 1

            X_seq.append(values[start:end])
            end_indices.append(idx_values[i])
            seq_active.append(active_values[start:end].any())

    if len(X_seq) == 0:
        return (
            np.empty((0, seq_len, feature_df.shape[1])),
            pd.Index([]),
            np.array([], dtype=bool)
        )

    return np.asarray(X_seq), pd.Index(end_indices), np.asarray(seq_active)


def reconstruction_error(model, X):
    recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=(1, 2))


def feature_reconstruction_error(model, X, feature_names):
    recon = model.predict(X, verbose=0)
    err = np.mean(np.square(X - recon), axis=1)
    return pd.DataFrame(err, columns=feature_names)


# ============================================================
# STEP 1: PREPARE GROUP DATAFRAMES
# Uses df_filled_dict from your preprocessing
# ============================================================

group_pdfs = {}
group_scalers = {}
group_activity = {}
group_feature_names = {}

for group, pdf in df_filled_dict.items():
    print(f"Preparing group: {group}")

    pdf = pdf.copy()

    if "bucket_start_unix" not in pdf.columns:
        raise ValueError(f"Group '{group}' must contain bucket_start_unix")

    if pdf.index.name != "lift_group_bucket":
        if "lift_group_bucket" in pdf.columns:
            pdf = pdf.set_index("lift_group_bucket")
        else:
            raise ValueError(f"Group '{group}' must have index or column lift_group_bucket")

    pdf = pdf.sort_values("bucket_start_unix")

    feature_df = pdf.drop(columns=["bucket_start_unix"], errors="ignore").copy()
    feature_df.columns = [c.replace("ImportS7Device.", "") for c in feature_df.columns]
    feature_df = feature_df.select_dtypes(include=[np.number])

    if feature_df.shape[1] == 0:
        print(f"  Skipping group '{group}' because it has no numeric columns.")
        continue

    feature_df = feature_df.fillna(feature_df.median())

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

    print(f"  Final shape: {feature_scaled.shape}")
    print(f"  Active rows: {activity_mask.sum()} / {len(activity_mask)}")


# ============================================================
# STEP 2: TRAIN ONE LSTM AUTOENCODER SPECIALIST PER GROUP
# ============================================================

specialists = {}
specialist_scores = {}
specialist_thresholds = {}
specialist_seq_lengths = {}

for group, pdf in group_pdfs.items():
    print("\n" + "=" * 60)
    print(f"Training LSTM autoencoder specialist: {group}")
    print("=" * 60)

    seq_len = sequence_lengths.get(group, 12)
    activity_mask = group_activity[group]

    X_seq, end_indices, seq_active = build_sequences_per_lift(
        pdf,
        activity_mask,
        seq_len
    )

    if len(X_seq) == 0:
        print(f"  Skipping group '{group}' because no sequences were created.")
        continue

    X_train = X_seq[seq_active]

    if len(X_train) < 50:
        print(f"  Too few active sequences ({len(X_train)}), training on all sequences.")
        X_train = X_seq

    n_features = X_seq.shape[2]

    model = make_lstm_autoencoder(
        seq_len=seq_len,
        n_features=n_features
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15 if len(X_train) > 100 else 0.0,
        callbacks=[early_stop] if len(X_train) > 100 else [],
        verbose=1
    )

    errors = reconstruction_error(model, X_seq)

    active_errors = errors[seq_active] if seq_active.sum() > 0 else errors
    threshold = robust_upper_threshold(active_errors, method="iqr", factor=3.0)

    pred = np.where(
        seq_active & (errors > threshold),
        -1,
        1
    )

    specialists[group] = model
    specialist_thresholds[group] = float(threshold)
    specialist_seq_lengths[group] = int(seq_len)

    specialist_scores[group] = pd.DataFrame({
        f"{group}_score": errors,
        f"{group}_pred": pred,
        f"{group}_active": seq_active.astype(int),
        "bucket_start_unix": [
            parse_lift_group_bucket(idx)[1] for idx in end_indices
        ]
    }, index=end_indices)

    print(f"  Sequences: {len(X_seq)}")
    print(f"  Active sequences: {seq_active.sum()} / {len(seq_active)}")
    print(f"  Reconstruction threshold: {threshold:.8f}")
    print(f"  Anomalies found: {(pred == -1).sum()} / {len(pred)}")


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

    specialist_with_lift[group] = temp.reset_index().rename(
        columns={"index": "lift_group_bucket"}
    )


# ============================================================
# STEP 4: ALIGN ALL SPECIALISTS ONTO COMMON BASE TIMELINE
# ============================================================

base_group = min(bucket_sizes, key=bucket_sizes.get)

if base_group not in specialist_with_lift:
    base_group = sorted(specialist_with_lift.keys())[0]

print(f"\nBase timeline group: {base_group}")

meta_features = specialist_with_lift[base_group][[
    "lift_group_bucket",
    "lift_id",
    "timestamp",
    f"{base_group}_score",
    f"{base_group}_pred",
    f"{base_group}_active"
]].copy()

meta_features = meta_features.sort_values(
    ["lift_id", "timestamp"]
).reset_index(drop=True)

for group, scores_df in specialist_with_lift.items():
    if group == base_group:
        continue

    print(f"Merging group: {group}")

    right = scores_df[[
        "lift_id",
        "timestamp",
        f"{group}_score",
        f"{group}_pred",
        f"{group}_active"
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

meta_features = meta_features[
    meta_features["n_active_groups"] > 0
].copy().reset_index(drop=True)

print(f"\nFinal meta feature matrix: {meta_features.shape}")
print(f"Unique lifts: {meta_features['lift_id'].nunique()}")


# ============================================================
# STEP 5: TRAIN META AUTOENCODER
# ============================================================
# ============================================================
# STEP 5: TRAIN META AUTOENCODER - FIXED
# ============================================================

score_columns = [c for c in meta_features.columns if c.endswith("_score")]
pred_columns = [c for c in meta_features.columns if c.endswith("_pred")]
meta_input_cols = score_columns + pred_columns

X_meta_df = meta_features[meta_input_cols].copy()
X_meta_df = X_meta_df.fillna(X_meta_df.median())

# IMPORTANT:
# build_sequences_per_lift expects index = liftid_bucketunix
meta_sequence_index = (
    meta_features["lift_id"].astype(str)
    + "_"
    + (
        meta_features["timestamp"].astype("int64") // 10**9
    ).astype(str)
)

X_meta_df.index = meta_sequence_index

meta_activity_mask = pd.Series(
    meta_features["n_active_groups"].values > 0,
    index=meta_sequence_index
)

meta_scaler = StandardScaler()

X_meta_scaled_df = pd.DataFrame(
    meta_scaler.fit_transform(X_meta_df),
    index=meta_sequence_index,
    columns=X_meta_df.columns
)

META_SEQ_LEN = 12

X_meta_seq, meta_end_indices, meta_seq_active = build_sequences_per_lift(
    X_meta_scaled_df,
    meta_activity_mask,
    META_SEQ_LEN
)

print("Meta sequences created:", len(X_meta_seq))

if len(X_meta_seq) == 0:
    raise ValueError(
        "No meta sequences created. Try lowering META_SEQ_LEN to 3 or check lift grouping."
    )

X_meta_train = X_meta_seq[meta_seq_active]

if len(X_meta_train) < 50:
    X_meta_train = X_meta_seq

meta_model = make_lstm_autoencoder(
    seq_len=META_SEQ_LEN,
    n_features=X_meta_seq.shape[2]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

meta_model.fit(
    X_meta_train,
    X_meta_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15 if len(X_meta_train) > 100 else 0.0,
    callbacks=[early_stop] if len(X_meta_train) > 100 else [],
    verbose=1
)

meta_errors = reconstruction_error(meta_model, X_meta_seq)

active_meta_errors = meta_errors[meta_seq_active] if meta_seq_active.sum() > 0 else meta_errors

meta_threshold = robust_upper_threshold(
    active_meta_errors,
    method="iqr",
    factor=3.0
)

meta_score_df = pd.DataFrame({
    "meta_score": meta_errors,
    "meta_pred": np.where(
        meta_seq_active & (meta_errors > meta_threshold),
        -1,
        1
    )
}, index=meta_end_indices)

meta_features = meta_features.copy()
meta_features["_meta_sequence_index"] = meta_sequence_index

meta_features["meta_score"] = np.nan
meta_features["meta_pred"] = 1

meta_features = meta_features.set_index("_meta_sequence_index", drop=False)

meta_features.loc[meta_score_df.index, "meta_score"] = meta_score_df["meta_score"]
meta_features.loc[meta_score_df.index, "meta_pred"] = meta_score_df["meta_pred"]

meta_features = meta_features.dropna(subset=["meta_score"]).copy()
meta_features = meta_features.reset_index(drop=True)

n_meta_anomalies = int((meta_features["meta_pred"] == -1).sum())

print(f"\nMeta reconstruction threshold: {meta_threshold:.8f}")
print(f"Meta autoencoder anomalies: {n_meta_anomalies} / {len(meta_features)}")

# ============================================================
# STEP 6: RESULTS PER LIFT
# ============================================================

anomaly_summary = (
    meta_features
    .groupby("lift_id")
    .agg(
        total_samples=("meta_pred", "count"),
        anomalies=("meta_pred", lambda x: (x == -1).sum()),
        worst_score=("meta_score", "max"),
        mean_score=("meta_score", "mean"),
        active_groups_mean=("n_active_groups", "mean")
    )
)

anomaly_summary["anomaly_pct"] = (
    100 * anomaly_summary["anomalies"] / anomaly_summary["total_samples"]
)

anomaly_summary = anomaly_summary.sort_values(
    ["anomalies", "worst_score"],
    ascending=[False, False]
)

print("\n" + "=" * 60)
print("ANOMALIES PER LIFT - LSTM AUTOENCODER")
print("=" * 60)
print(anomaly_summary.to_string())


# ============================================================
# STEP 7: JOIN MOVEMENT TYPE
# ============================================================

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


# ============================================================
# STEP 8: VISUALIZE WORST LIFT
# ============================================================

if len(anomaly_summary) > 0:
    worst_lift = anomaly_summary.index[0]

    lift_data = meta_features[
        meta_features["lift_id"].astype(str) == str(worst_lift)
    ].copy().sort_values("timestamp")

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

        ax.plot(
            lift_data["timestamp"],
            lift_data[col],
            linewidth=0.7,
            alpha=0.8
        )

        ax.scatter(
            lift_data.loc[anomaly_mask, "timestamp"],
            lift_data.loc[anomaly_mask, col],
            c="red",
            s=10,
            alpha=0.7
        )

        ax.set_ylabel(col.replace("_score", ""), fontsize=8)

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

    ax.axhline(meta_threshold, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("META ERROR", fontsize=10, fontweight="bold")

    plt.suptitle(f"LIFT {worst_lift} - Stacked LSTM Autoencoder Detection", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print(f"WHAT DRIVES ANOMALIES ON LIFT {worst_lift}?")
    print("=" * 60)

    for col in score_columns:
        normal_mean = lift_data.loc[~anomaly_mask, col].mean()
        anomaly_mean = lift_data.loc[anomaly_mask, col].mean()

        print(
            f"{col:40s} "
            f"normal={normal_mean:.6f} "
            f"anomaly={anomaly_mean:.6f}"
        )


# ============================================================
# STEP 9: SAVE ALL MODELS
# ============================================================
# ============================================================
# STEP 9: SAVE ALL MODELS - DATABRICKS SAFE
# ============================================================

import shutil

crane_files_dir = "/Volumes/craneds_dev/maxedge_lhdp/crane_files"
MODEL_DIR = os.path.join(crane_files_dir, "anomaly_model_lstm_autoencoder")

LOCAL_MODEL_DIR = "/tmp/anomaly_model_lstm_autoencoder"

if os.path.exists(LOCAL_MODEL_DIR):
    shutil.rmtree(LOCAL_MODEL_DIR)

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\nSaving locally first to: {LOCAL_MODEL_DIR}")
print("=" * 60)

config = {
    "model_type": "stacked_lstm_autoencoder",
    "bucket_sizes": bucket_sizes,
    "activity_thresholds": activity_thresholds,
    "specialist_thresholds": specialist_thresholds,
    "specialist_seq_lengths": specialist_seq_lengths,
    "meta_threshold": float(meta_threshold),
    "meta_seq_len": META_SEQ_LEN,
    "base_group": base_group,
    "score_columns": score_columns,
    "pred_columns": pred_columns,
    "group_feature_names": group_feature_names
}

with open(os.path.join(LOCAL_MODEL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print("  saved config.json")

for group, scaler in group_scalers.items():
    joblib.dump(
        scaler,
        os.path.join(LOCAL_MODEL_DIR, f"scaler_{group}.joblib")
    )
    print(f"  saved scaler_{group}.joblib")

for group, model in specialists.items():
    local_path = os.path.join(LOCAL_MODEL_DIR, f"specialist_{group}.keras")
    model.save(local_path)
    print(f"  saved specialist_{group}.keras")

joblib.dump(
    meta_scaler,
    os.path.join(LOCAL_MODEL_DIR, "meta_scaler.joblib")
)

meta_model.save(
    os.path.join(LOCAL_MODEL_DIR, "meta_model.keras")
)

print("  saved meta_scaler.joblib")
print("  saved meta_model.keras")

# Copy files from local disk to Volume
print(f"\nCopying saved files to Volume: {MODEL_DIR}")
print("=" * 60)

for file_name in os.listdir(LOCAL_MODEL_DIR):
    src = os.path.join(LOCAL_MODEL_DIR, file_name)
    dst = os.path.join(MODEL_DIR, file_name)

    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

print("\nSaved files:")
total_size = 0

for file_name in sorted(os.listdir(MODEL_DIR)):
    file_path = os.path.join(MODEL_DIR, file_name)
    size_kb = os.path.getsize(file_path) / 1024
    total_size += size_kb
    print(f"  {file_name:45s} {size_kb:10.1f} KB")

print(f"  {'TOTAL':45s} {total_size:10.1f} KB")
print(f"\nFinal model path: {MODEL_DIR}")