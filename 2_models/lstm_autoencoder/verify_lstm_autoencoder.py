# Databricks notebook source
# MAGIC %pip install tensorflow
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# ============================================================
# FIXED LSTM ANOMALY SCORING SCRIPT - PART A
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when
from itertools import chain


# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/anomaly_model_lstm_autoencoder"
TIMESTAMP_COL = "timestamp_utc"

# ============================================================
# PART 1: LOAD TRAINED LSTM ARTIFACTS
# ============================================================
with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)

bucket_sizes = config["bucket_sizes"]
activity_thresholds = config.get("activity_thresholds", {})
specialist_thresholds = config["specialist_thresholds"]
specialist_seq_lengths = config["specialist_seq_lengths"]
meta_threshold = config["meta_threshold"]
meta_seq_len = config["meta_seq_len"]
base_group = config["base_group"]
group_feature_names = config["group_feature_names"]

group_scalers = {}
specialists = {}
for group in group_feature_names.keys():
    sp = os.path.join(MODEL_DIR, f"scaler_{group}.joblib")
    mp = os.path.join(MODEL_DIR, f"specialist_{group}.keras")
    if os.path.exists(sp):
        group_scalers[group] = joblib.load(sp)
    if os.path.exists(mp):
        specialists[group] = load_model(mp)

meta_scaler = joblib.load(os.path.join(MODEL_DIR, "meta_scaler.joblib"))
meta_model = load_model(os.path.join(MODEL_DIR, "meta_model.keras"))

print("=" * 60)
print("LOADED TRAINED LSTM ARTIFACTS")
print("=" * 60)
print("Groups:", list(group_feature_names.keys()))
print("Base group:", base_group)
print("Meta threshold:", meta_threshold)
print("Specialist seq_lengths:", specialist_seq_lengths)
print("Meta seq_len:", meta_seq_len)


# ============================================================
# PART 2: LOAD NEW CRANE DATA
# ============================================================
df_liftlog = spark.table("default.df_liftlog_overload")
df_liftlog = df_liftlog.filter(col("movementtype") != 5)

valid_lift_ids = df_liftlog.select("lift_id").distinct()

df_clean_with_lift = (
    spark.table("default.df_raw_liftid_overload")
        .join(valid_lift_ids, on="lift_id", how="inner")
)

n_input_lifts = df_clean_with_lift.select("lift_id").distinct().count()
print(f"\nLifts entering pipeline (after movementtype filter): {n_input_lifts}")


# ============================================================
# PART 5: TAG GROUP DEFINITIONS
# ============================================================
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
    "ImportS7Device.Encoders.BoomA_Raw","ImportS7Device.Encoders.BoomB_Raw",
    "ImportS7Device.General.BoomAngle","ImportS7Device.General.BoomRadius",
    "ImportS7Device.Drives.Boom_Status.ActualSpeed","ImportS7Device.Drives.Boom_Status.ActualPower",
    "ImportS7Device.Drives.Boom_Control.SpeedReference","ImportS7Device.Joysticks.Boom",
    "ImportS7Device.Drives.Boom_Status.ActualCurrent","ImportS7Device.Drives.Boom_Status.ActualTorque",
    "ImportS7Device.Hyd.BoomPrimaryPressure",
]
tagnames_hoist_active = [
    "ImportS7Device.Drives.Hoist_Status.ActualCurrent","ImportS7Device.Joysticks.Hoist",
    "ImportS7Device.Drives.Hoist_Status.ActualSpeed","ImportS7Device.Drives.Hoist_Control.SpeedReference",
    "ImportS7Device.Encoders.MainA_Raw","ImportS7Device.Encoders.MainB_Raw",
    "ImportS7Device.Drives.Hoist_Status.ActualPower","ImportS7Device.General.HoistPosition",
]
tagnames_hyd_common = [
    "ImportS7Device.Hyd.BrakeSystemPressure","ImportS7Device.Hyd.BrakeAccumulatorPressure",
]
tagnames_slew_active = [
    "ImportS7Device.Encoders.Slew_Raw","ImportS7Device.General.SlewAngle","ImportS7Device.Joysticks.Slew",
    "ImportS7Device.Drives.SlewA_Control.SpeedReference","ImportS7Device.Drives.SlewB_Control.SpeedReference",
    "ImportS7Device.Drives.SlewA_Status.ActualSpeed","ImportS7Device.Drives.SlewB_Status.ActualSpeed",
    "ImportS7Device.Drives.SlewB_Status.ActualTorque","ImportS7Device.Drives.SlewB_Status.ActualCurrent",
    "ImportS7Device.Drives.SlewB_Status.DroopFeedbackSpeedReduction",
    "ImportS7Device.Drives.SlewA_Status.ActualTorque",
    "ImportS7Device.Drives.SlewA_Status.DroopFeedbackSpeedReduction",
    "ImportS7Device.Drives.SlewA_Status.ActualCurrent",
    "ImportS7Device.CBM.SlewGear_Moment","ImportS7Device.General.SlewMoment",
    "ImportS7Device.Hyd.SlewBrakePressureA","ImportS7Device.Hyd.SlewBrakePressureB",
]
tagnames_slew_power = [
    "ImportS7Device.Drives.SlewA_Status.ActualPower","ImportS7Device.Drives.SlewB_Status.ActualPower",
    "ImportS7Device.Drives.SlewB_Status.TorqueLimEffective","ImportS7Device.Drives.SlewA_Status.TorqueLimEffective",
]
tagnames_thermal_drive = [
    "ImportS7Device.Drives.Hoist_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.SlewB_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.Boom_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.SlewA_Status.DepletionLayerMaxTemp",
]
tagnames_thermal_motor = [
    "ImportS7Device.Drives.Hoist_Status.MotorTemp","ImportS7Device.Drives.Boom_Status.MotorTemp",
    "ImportS7Device.Drives.SlewA_Status.MotorTemp","ImportS7Device.Drives.SlewB_Status.MotorTemp",
]
tagnames_thermal_resistor = [
    "ImportS7Device.HVAC.ResistorCoolant.BrakeResistorTempA",
    "ImportS7Device.HVAC.ResistorCoolant.BrakeResistorTempB",
    "ImportS7Device.HVAC.ResistorCoolant.ResistorTankCoolantTemp",
]

tagnames = (
    tagnames_always_on + tagnames_boom_active + tagnames_hoist_active +
    tagnames_hyd_common + tagnames_slew_active + tagnames_slew_power +
    tagnames_thermal_drive + tagnames_thermal_motor + tagnames_thermal_resistor
)

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

taggroups = {
    "always_on": tagnames_always_on, "boom_active": tagnames_boom_active,
    "hoist_active": tagnames_hoist_active, "hyd_common": tagnames_hyd_common,
    "slew_active": tagnames_slew_active, "slew_power": tagnames_slew_power,
    "thermal_drive": tagnames_thermal_drive, "thermal_motor": tagnames_thermal_motor,
    "thermal_resistor": tagnames_thermal_resistor,
}


# ============================================================
# PART 6: PREPROCESS NEW DATA (BUCKETIZE + PIVOT)
# ============================================================
df_raw_selected = (
    df_clean_with_lift
    .filter(F.col("tagname").isin(tagnames))
    .withColumn("tag_group", tag_group_expr)
    .withColumn(TIMESTAMP_COL, F.col(TIMESTAMP_COL).cast("timestamp"))
)

bucket_map_expr = F.create_map([F.lit(x) for x in chain(*bucket_sizes.items())])

df_raw_buckets = (
    df_raw_selected
    .withColumn("bucket_size_s", bucket_map_expr[F.col("tag_group")].cast("int"))
    .withColumn("ts_unix", F.unix_timestamp(F.col(TIMESTAMP_COL)))
    .withColumn(
        "bucket_start_unix",
        (F.floor(F.col("ts_unix") / F.col("bucket_size_s")) * F.col("bucket_size_s")).cast("long")
    )
    .withColumn(
        "lift_group_bucket",
        F.concat_ws("_", F.col("lift_id").cast("string"), F.col("bucket_start_unix").cast("string"))
    )
)

df_pivoted_dict_new = {}
for group, tags in taggroups.items():
    df_group = df_raw_buckets.filter(F.col("tag_group") == group)
    if df_group.limit(1).count() > 0:
        df_pivoted_dict_new[group] = (
            df_group
            .groupBy("lift_group_bucket", "bucket_start_unix")
            .pivot("tagname", tags)
            .agg(F.mean("value"))
            .orderBy("lift_group_bucket")
        )

print("\nAvailable groups in new data:")
for g in df_pivoted_dict_new:
    print(" -", g)


# ============================================================
# PART 7: FILLING LOGIC
# ============================================================
def weighted_fill_series(values, max_gap=10, zero_gap_fill=True):
    filled = values.astype(float).copy()
    n = len(filled)
    i = 0
    while i < n:
        if not np.isnan(filled[i]):
            i += 1; continue
        start = i
        while i < n and np.isnan(filled[i]):
            i += 1
        end = i - 1
        left_idx = start - 1
        right_idx = end + 1
        left_valid = left_idx >= 0 and not np.isnan(filled[left_idx])
        right_valid = right_idx < n and not np.isnan(filled[right_idx])
        gap_length = end - start + 1
        if (zero_gap_fill and left_valid and right_valid and
                filled[left_idx] == 0 and filled[right_idx] == 0):
            filled[start:end + 1] = 0
            continue
        if gap_length <= max_gap:
            for k in range(start, end + 1):
                neighbors, weights = [], []
                if left_valid:
                    neighbors.append(filled[left_idx]); weights.append(1.0/(k-left_idx))
                if right_valid:
                    neighbors.append(filled[right_idx]); weights.append(1.0/(right_idx-k))
                if weights:
                    filled[k] = np.dot(neighbors, weights) / np.sum(weights)
    return filled

def extract_lift_id(index_series):
    return index_series.astype(str).str.extract(r"^(\d+)_")[0]

fill_windows = {
    "always_on": 6, "boom_active": 6, "hoist_active": 6, "hyd_common": 4,
    "slew_active": 6, "slew_power": 2, "thermal_drive": 3,
    "thermal_motor": 2, "thermal_resistor": 2,
}

zero_fill_columns = {
    "ImportS7Device.CBM.LuffingWinch_Moment",
    "ImportS7Device.CBM.MainHoistWinch_Moment",
    "ImportS7Device.CBM.SlewBearing_Moment",
    "ImportS7Device.General.BoomLoad",
    "ImportS7Device.General.HoistLoadPctSWL",
    "ImportS7Device.General.PedestalMoment",
    "ImportS7Device.SafetySystems.AOPS RopeForce",
    "ImportS7Device.Drives.Boom_Status.ActualSpeed",
    "ImportS7Device.Drives.Boom_Status.ActualPower",
    "ImportS7Device.Drives.Boom_Status.ActualCurrent",
    "ImportS7Device.Drives.Boom_Status.ActualTorque",
    "ImportS7Device.Drives.Boom_Control.SpeedReference",
    "ImportS7Device.Joysticks.Boom",
    "ImportS7Device.Drives.Hoist_Status.ActualSpeed",
    "ImportS7Device.Drives.Hoist_Status.ActualPower",
    "ImportS7Device.Drives.Hoist_Status.ActualCurrent",
    "ImportS7Device.Drives.Hoist_Control.SpeedReference",
    "ImportS7Device.Joysticks.Hoist",
    "ImportS7Device.Drives.Hoist_Status.ActualTorque",
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
    "ImportS7Device.Drives.SlewA_Status.ActualPower",
    "ImportS7Device.Drives.SlewB_Status.ActualPower",
    "ImportS7Device.Drives.SlewA_Status.TorqueLimEffective",
    "ImportS7Device.Drives.SlewB_Status.TorqueLimEffective",
}

df_filled_dict_new = {}

for group, spark_df in df_pivoted_dict_new.items():
    print(f"\nFilling new data group: {group}")

    pdf = spark_df.toPandas()
    pdf = pdf.sort_values(["lift_group_bucket", "bucket_start_unix"]).copy()
    pdf = pdf.set_index("lift_group_bucket", drop=True)
    pdf["_lift_id"] = extract_lift_id(pdf.index.to_series())

    data_cols = [c for c in pdf.columns if c not in ["bucket_start_unix", "_lift_id"]]
    max_gap = fill_windows.get(group, 5)

    filled_parts = []
    for lift_id, subdf in pdf.groupby("_lift_id", sort=False):
        subdf = subdf.sort_values("bucket_start_unix").copy()
        for c in data_cols:
            subdf[c] = weighted_fill_series(
                subdf[c].values,
                max_gap=max_gap,
                zero_gap_fill=(c in zero_fill_columns)
            )
        filled_parts.append(subdf)

    pdf_filled = pd.concat(filled_parts).sort_values(["_lift_id", "bucket_start_unix"])
    pdf_filled = pdf_filled.drop(columns=["_lift_id"])
    df_filled_dict_new[group] = pdf_filled


# ============================================================
# PART 8: SCORE NEW DATA WITH TRAINED LSTM MODELS
# ============================================================

def compute_row_activity(pdf, threshold=0.3):
    if pdf.shape[1] == 0:
        return pd.Series(False, index=pdf.index)
    row_max = np.abs(pdf.values).max(axis=1)
    return pd.Series(row_max > threshold, index=pdf.index)


def parse_lift_group_bucket(index_val):
    parts = str(index_val).split("_")
    return parts[0], int(parts[-1])


def build_lstm_prediction_windows(feature_df, activity_mask, seq_len,
                                  pad_short=True, group_label=""):
    """
    Forecasting windows:
       X = previous seq_len rows, y = current row
    If pad_short=True, lifts with len <= seq_len are padded by repeating the
    first row so they still produce >=1 window (padded rows are marked inactive).
    """
    X, y, end_indices, active_out = [], [], [], []

    temp = feature_df.copy()
    temp["_lift_id"]     = [parse_lift_group_bucket(idx)[0] for idx in temp.index]
    temp["_bucket_unix"] = [parse_lift_group_bucket(idx)[1] for idx in temp.index]
    temp["_active"]      = activity_mask.astype(bool).values

    feature_cols = [c for c in temp.columns
                    if c not in ["_lift_id", "_bucket_unix", "_active"]]

    n_lifts_total   = 0
    n_lifts_padded  = 0
    n_lifts_dropped = 0

    for lift_id, subdf in temp.groupby("_lift_id", sort=False):
        n_lifts_total += 1
        subdf = subdf.sort_values("_bucket_unix")

        values        = subdf[feature_cols].values
        active_values = subdf["_active"].values
        idx_values    = subdf.index.to_numpy()

        if len(subdf) <= seq_len:
            if not pad_short or len(subdf) == 0:
                n_lifts_dropped += 1
                continue
            pad_n = seq_len + 1 - len(subdf)
            values        = np.vstack([np.repeat(values[:1], pad_n, axis=0), values])
            active_values = np.concatenate([np.zeros(pad_n, dtype=bool), active_values])
            idx_values    = np.concatenate([np.repeat(idx_values[:1], pad_n), idx_values])
            n_lifts_padded += 1

        for i in range(seq_len, len(values)):
            X.append(values[i - seq_len:i])
            y.append(values[i])
            end_indices.append(idx_values[i])
            # require the predicted row OR any row in the window to be active
            active_out.append(active_values[i - seq_len:i + 1].any())

    print(f"  [{group_label}] total lifts seen: {n_lifts_total}")
    print(f"  [{group_label}] lifts padded (len<=seq_len={seq_len}): {n_lifts_padded}")
    print(f"  [{group_label}] lifts dropped: {n_lifts_dropped}")

    if len(X) == 0:
        return (np.empty((0, seq_len, feature_df.shape[1])),
                np.empty((0, feature_df.shape[1])),
                pd.Index([]), np.array([], dtype=bool))

    return (np.asarray(X), np.asarray(y),
            pd.Index(end_indices), np.asarray(active_out))


def lstm_error(model, X, y_true):
    """
    Computes per-window error.

    - If model outputs (batch, seq_len, features)  -> AUTOENCODER: MSE(X, y_pred)
    - If model outputs (batch, features)           -> FORECASTER:  MSE(y_true, y_pred)
    - If model outputs (batch, 1, features)        -> FORECASTER:  MSE(y_true, y_pred.squeeze)
    """
    y_pred = model.predict(X, verbose=0)

    # Autoencoder: full window reconstruction
    if y_pred.ndim == 3 and y_pred.shape[1] == X.shape[1]:
        return np.mean(np.square(X - y_pred), axis=(1, 2))

    # Forecaster with shape (batch, 1, features)
    if y_pred.ndim == 3 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(1)
        return np.mean(np.square(y_true - y_pred), axis=1)

    # Forecaster with shape (batch, features)
    if y_pred.ndim == 2:
        return np.mean(np.square(y_true - y_pred), axis=1)

    raise ValueError(f"Unexpected model output shape: {y_pred.shape}")


specialist_scores_new = {}
specialist_with_lift_new = {}

for group, pdf in df_filled_dict_new.items():
    if group not in group_scalers or group not in specialists:
        print(f"Skipping group '{group}' because LSTM model/scaler not found.")
        continue

    print(f"\nScoring new data group with LSTM: {group}")

    pdf = pdf.copy()
    if "bucket_start_unix" not in pdf.columns:
        raise ValueError(f"Group '{group}' missing bucket_start_unix")

    feature_df = pdf.drop(columns=["bucket_start_unix"], errors="ignore").copy()
    feature_df.columns = [c.replace("ImportS7Device.", "") for c in feature_df.columns]
    feature_df = feature_df.select_dtypes(include=[np.number])

    expected_cols = group_feature_names[group]
    for c in expected_cols:
        if c not in feature_df.columns:
            feature_df[c] = np.nan
    feature_df = feature_df[expected_cols]
    feature_df = feature_df.fillna(feature_df.median())

    scaler = group_scalers[group]
    model = specialists[group]

    feature_scaled = pd.DataFrame(
        scaler.transform(feature_df),
        index=feature_df.index,
        columns=feature_df.columns
    )

    activity_mask = compute_row_activity(
        feature_scaled,
        threshold=activity_thresholds.get(group, 0.3)
    )

    seq_len = specialist_seq_lengths[group]

    X_seq, y_true, end_indices, seq_active = build_lstm_prediction_windows(
        feature_scaled, activity_mask, seq_len,
        pad_short=True, group_label=group
    )

    if len(X_seq) == 0:
        print(f"  No valid LSTM windows for group '{group}'")
        continue

    errors = lstm_error(model, X_seq, y_true)
    threshold = specialist_thresholds[group]

    pred = np.where(seq_active & (errors > threshold), -1, 1)

    scored_df = pd.DataFrame({
        f"{group}_score": errors,
        f"{group}_pred": pred,
        f"{group}_active": seq_active.astype(int),
        "bucket_start_unix": [parse_lift_group_bucket(idx)[1] for idx in end_indices]
    }, index=end_indices)

    specialist_scores_new[group] = scored_df

    parsed = [parse_lift_group_bucket(idx) for idx in scored_df.index]
    temp = scored_df.copy()
    temp["lift_id"]   = [p[0] for p in parsed]
    temp["bucket_unix"] = [p[1] for p in parsed]
    temp["timestamp"] = pd.to_datetime(temp["bucket_unix"], unit="s", utc=True)

    specialist_with_lift_new[group] = temp.reset_index().rename(
        columns={"index": "lift_group_bucket"}
    )

    print(f"  LSTM windows: {len(X_seq)}")
    print(f"  Active windows: {seq_active.sum()} / {len(seq_active)}")
    print(f"  Threshold used: {threshold:.8f}")
    print(f"  Anomalies found: {(pred == -1).sum()} / {len(pred)}")
    print(f"  Unique lifts after windowing: {temp['lift_id'].nunique()}")


# ============================================================
# PART 9: ALIGN SPECIALISTS ONTO COMMON TIMELINE
#         (timeline = union of all lifts across specialists,
#          NOT just the base group)
# ============================================================

# Build a "skeleton" timeline from the union of all (lift_id, timestamp)
# pairs across specialists, so no lift is silently lost just because it
# isn't in the base group.
skeleton_parts = []
for g, df_s in specialist_with_lift_new.items():
    skeleton_parts.append(df_s[["lift_id", "timestamp"]])

skeleton = (
    pd.concat(skeleton_parts, ignore_index=True)
    .drop_duplicates(subset=["lift_id", "timestamp"])
    .sort_values(["lift_id", "timestamp"])
    .reset_index(drop=True)
)
print(f"\nSkeleton (union) timeline rows: {len(skeleton)}, "
      f"unique lifts: {skeleton['lift_id'].nunique()}")

# Use base_group columns as the "primary" specialist if available;
# otherwise fall back to first available group.
if base_group not in specialist_with_lift_new:
    base_group = sorted(specialist_with_lift_new.keys())[0]
    print(f"Base group missing; using fallback base group: {base_group}")

meta_features_new = skeleton.copy()

# Merge each specialist (including base_group) onto the skeleton,
# per-lift, using merge_asof (backward).
for group, scores_df in specialist_with_lift_new.items():
    print(f"Merging specialist group into meta timeline: {group}")

    right = scores_df[[
        "lift_id", "timestamp",
        f"{group}_score", f"{group}_pred", f"{group}_active"
    ]].sort_values(["lift_id", "timestamp"]).reset_index(drop=True)

    merged_parts = []
    for lid in meta_features_new["lift_id"].unique():
        left_lift  = meta_features_new[meta_features_new["lift_id"] == lid].copy()
        right_lift = right[right["lift_id"] == lid].copy()

        if len(right_lift) == 0:
            left_lift[f"{group}_score"]  = np.nan
            left_lift[f"{group}_pred"]   = np.nan
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

    meta_features_new = pd.concat(merged_parts, ignore_index=True)
    print(f"  Shape after merge: {meta_features_new.shape}")

active_cols = [c for c in meta_features_new.columns if c.endswith("_active")]
meta_features_new["n_active_groups"] = meta_features_new[active_cols].fillna(0).sum(axis=1)

# Keep rows with at least one active subsystem
meta_features_new = meta_features_new[
    meta_features_new["n_active_groups"] > 0
].copy().reset_index(drop=True)

print("\nFinal meta feature table shape:", meta_features_new.shape)
print("Unique lifts in meta table:", meta_features_new["lift_id"].nunique())


# ============================================================
# PART 10: SCORE META LSTM MODEL
# ============================================================

score_columns = [c for c in meta_features_new.columns if c.endswith("_score")]
pred_columns  = [c for c in meta_features_new.columns if c.endswith("_pred")]
meta_input_cols = score_columns + pred_columns

X_meta_df_new = meta_features_new[meta_input_cols].copy()
X_meta_df_new = X_meta_df_new.fillna(X_meta_df_new.median())

# Build a stable index "lift_id_unixts" to use with build_lstm_prediction_windows
meta_sequence_index = (
    meta_features_new["lift_id"].astype(str)
    + "_"
    + (meta_features_new["timestamp"].astype("int64") // 10**9).astype(str)
)
X_meta_df_new.index = meta_sequence_index

X_meta_scaled_df_new = pd.DataFrame(
    meta_scaler.transform(X_meta_df_new),
    index=meta_sequence_index,
    columns=X_meta_df_new.columns
)

meta_activity_mask_new = pd.Series(
    meta_features_new["n_active_groups"].values > 0,
    index=meta_sequence_index
)

print("\nBuilding META LSTM windows ...")
X_meta_seq_new, y_meta_true_new, meta_end_indices_new, meta_seq_active_new = build_lstm_prediction_windows(
    X_meta_scaled_df_new,
    meta_activity_mask_new,
    meta_seq_len,
    pad_short=True,
    group_label="meta"
)

if len(X_meta_seq_new) == 0:
    raise ValueError("No meta LSTM windows created. Try lower meta_seq_len.")

meta_errors_new = lstm_error(meta_model, X_meta_seq_new, y_meta_true_new)
meta_score_df = pd.DataFrame({
    "meta_score": meta_errors_new,
    "meta_pred": np.where(
        meta_seq_active_new & (meta_errors_new > meta_threshold),
        -1, 1
    )
}, index=meta_end_indices_new)

# Map meta scores back onto meta_features_new via the sequence index
meta_features_new["_meta_sequence_index"] = meta_sequence_index
meta_features_new = meta_features_new.set_index("_meta_sequence_index", drop=False)

meta_features_new["meta_score"] = np.nan
meta_features_new["meta_pred"] = 1

# Some sequence indices might be duplicated (same lift_id+timestamp from
# different specialists merged together). Drop dups before assigning.
meta_score_df = meta_score_df[~meta_score_df.index.duplicated(keep="last")]

valid_idx = meta_score_df.index.intersection(meta_features_new.index)
meta_features_new.loc[valid_idx, "meta_score"] = meta_score_df.loc[valid_idx, "meta_score"]
meta_features_new.loc[valid_idx, "meta_pred"]  = meta_score_df.loc[valid_idx, "meta_pred"]

meta_features_new = meta_features_new.dropna(subset=["meta_score"]).copy()
meta_features_new = meta_features_new.reset_index(drop=True)

n_meta_anom = int((meta_features_new["meta_pred"] == -1).sum())

print("\n" + "=" * 60)
print("LSTM META MODEL RESULTS ON STRESS-TEST DATA")
print("=" * 60)
print(f"Meta threshold: {meta_threshold:.8f}")
print(f"Meta anomalies found: {n_meta_anom} / {len(meta_features_new)}")
if len(meta_features_new) > 0:
    print(f"Anomaly rate: {100 * n_meta_anom / len(meta_features_new):.2f}%")


# ============================================================
# PART 11: SUMMARY PER LIFT
# ============================================================

anomaly_summary_new = (
    meta_features_new
    .groupby("lift_id")
    .agg(
        total_samples=("meta_pred", "count"),
        anomalies=("meta_pred", lambda x: (x == -1).sum()),
        worst_score=("meta_score", "max"),
        mean_score=("meta_score", "mean"),
        active_groups_mean=("n_active_groups", "mean")
    )
)

anomaly_summary_new["anomaly_pct"] = (
    100 * anomaly_summary_new["anomalies"] / anomaly_summary_new["total_samples"]
)

anomaly_summary_new = anomaly_summary_new.sort_values(
    ["anomalies", "worst_score"], ascending=[False, False]
)

print("\n" + "=" * 60)
print("ANOMALY SUMMARY PER LIFT - LSTM STRESS TEST")
print("=" * 60)
print(anomaly_summary_new.to_string())


# ============================================================
# PART 12: SPECIALIST ANOMALY RATES
# ============================================================

specialist_summary = []
for group, df_scores in specialist_scores_new.items():
    pred_col   = f"{group}_pred"
    active_col = f"{group}_active"

    total_rows  = len(df_scores)
    active_rows = int(df_scores[active_col].sum())
    anomalies   = int((df_scores[pred_col] == -1).sum())

    specialist_summary.append({
        "group": group,
        "total_rows": total_rows,
        "active_rows": active_rows,
        "anomalies": anomalies,
        "anomaly_rate_all_pct":    100 * anomalies / total_rows  if total_rows  else 0,
        "anomaly_rate_active_pct": 100 * anomalies / active_rows if active_rows else 0,
    })

specialist_summary_df = pd.DataFrame(specialist_summary).sort_values(
    "anomaly_rate_active_pct", ascending=False
)

print("\n" + "=" * 60)
print("SPECIALIST LSTM ANOMALY RATES")
print("=" * 60)
print(specialist_summary_df.to_string(index=False))


# ============================================================
# PART 13: DETECTION CHECK
# ============================================================
print("\n" + "=" * 60)
print("LSTM STRESS TEST DETECTION CHECK")
print("=" * 60)

if len(meta_features_new) == 0:
    print("No meta rows available after preprocessing/scoring.")
else:
    any_anomalies = (meta_features_new["meta_pred"] == -1).any()
    lifts_with_anomalies = int((anomaly_summary_new["anomalies"] > 0).sum())

    print(f"Any anomalies detected? {'YES' if any_anomalies else 'NO'}")
    print(f"Lifts with at least one anomaly: "
          f"{lifts_with_anomalies} / {len(anomaly_summary_new)}")


# ============================================================
# PART 14: TOP ANOMALOUS WINDOWS
# ============================================================
top_anomalies = (
    meta_features_new.loc[
        meta_features_new["meta_pred"] == -1,
        ["lift_id", "timestamp", "meta_score", "n_active_groups"]
    ]
    .sort_values("meta_score", ascending=False)
    .head(50)
)

print("\n" + "=" * 60)
print("TOP ANOMALOUS LSTM TIME WINDOWS")
print("=" * 60)
print(top_anomalies.to_string(index=False))


# ============================================================
# PART 15: OPTIONAL MOVEMENT TYPE JOIN
# ============================================================
if "movementtype" in df_liftlog.columns:
    liftlog_pd = df_liftlog.select("lift_id", "movementtype").distinct().toPandas()
    liftid_to_movementtype = dict(
        zip(liftlog_pd["lift_id"].astype(str), liftlog_pd["movementtype"])
    )

    anomaly_summary_new = anomaly_summary_new.copy()
    anomaly_summary_new.index = anomaly_summary_new.index.astype(str)
    anomaly_summary_new["movementtype"] = anomaly_summary_new.index.map(
        liftid_to_movementtype.get
    )

    print("\n" + "=" * 60)
    print("ANOMALY SUMMARY WITH MOVEMENT TYPE")
    print("=" * 60)
    print(anomaly_summary_new.to_string())


# ============================================================
# PART 16: PLOT WORST LIFT
# ============================================================
if len(anomaly_summary_new) > 0:
    worst_lift = anomaly_summary_new.index[0]

    lift_data = meta_features_new[
        meta_features_new["lift_id"].astype(str) == str(worst_lift)
    ].copy().sort_values("timestamp")

    anomaly_mask = lift_data["meta_pred"] == -1

    print(f"\nDriver summary for worst lift: {worst_lift}")
    for col_name in score_columns:
        pred_col = col_name.replace("_score", "_pred")
        if pred_col in lift_data.columns:
            print(
                f"{col_name:40s} "
                f"anom_windows={(lift_data[pred_col] == -1).sum():5d} "
                f"mean_score={lift_data[col_name].mean(): .6f} "
                f"max_score={lift_data[col_name].max(): .6f}"
            )

    fig, axes = plt.subplots(
        len(score_columns) + 1,
        1,
        figsize=(20, 3 * (len(score_columns) + 1)),
        sharex=True
    )
    if len(score_columns) == 1:
        axes = [axes]

    for i, col_name in enumerate(score_columns):
        ax = axes[i]
        ax.plot(lift_data["timestamp"], lift_data[col_name],
                linewidth=0.8, alpha=0.85)
        ax.scatter(
            lift_data.loc[anomaly_mask, "timestamp"],
            lift_data.loc[anomaly_mask, col_name],
            c="red", s=10, alpha=0.7
        )
        ax.set_ylabel(col_name.replace("_score", ""), fontsize=8)

    ax = axes[-1]
    ax.plot(lift_data["timestamp"], lift_data["meta_score"],
            linewidth=1.0, color="black")
    ax.scatter(
        lift_data.loc[anomaly_mask, "timestamp"],
        lift_data.loc[anomaly_mask, "meta_score"],
        c="red", s=10, alpha=0.7
    )
    ax.axhline(meta_threshold, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("META LSTM ERROR", fontsize=10, fontweight="bold")

    plt.suptitle(
        f"Stress-Test Crane: Lift {worst_lift} - LSTM Prediction Errors",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print(f"WHAT DRIVES LSTM ANOMALIES ON STRESS-TEST LIFT {worst_lift}?")
    print("=" * 60)
    print(f"Normal windows: {(~anomaly_mask).sum()}")
    print(f"Anomalous windows: {anomaly_mask.sum()}")

    for col_name in score_columns:
        normal_vals  = lift_data.loc[~anomaly_mask, col_name]
        anomaly_vals = lift_data.loc[ anomaly_mask, col_name]

        normal_mean  = normal_vals.mean()  if len(normal_vals)  else np.nan
        anomaly_mean = anomaly_vals.mean() if len(anomaly_vals) else np.nan

        print(
            f"{col_name:40s} "
            f"normal={normal_mean:.6f} "
            f"anomaly={anomaly_mean:.6f}"
        )
else:
    print("No lifts available for visualization.")