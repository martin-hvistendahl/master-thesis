# Databricks notebook source
df_liftlog = spark.table("default.df_liftlog_overload")
from pyspark.sql.functions import col, countDistinct as F
df_liftlog = df_liftlog.filter(col("movementtype") != 5)


display(
    df_liftlog.groupBy("movementtype")
    .agg(F.countDistinct("lift_id").alias("num_lifts"))
    .orderBy("movementtype")
)

# COMMAND ----------

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, coalesce
from itertools import chain


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/anomaly_model"
BASE_DIR = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/csv_categories/"
PREFIX = "EQ-25206"
TIMESTAMP_COL = "timestamp_utc"

# ============================================================
# PART 1: LOAD TRAINED MODEL ARTIFACTS
# ============================================================

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)

bucket_sizes = config["bucket_sizes"]
activity_thresholds = config.get("activity_thresholds", {})
specialist_thresholds = config.get("specialist_thresholds", {})
meta_threshold = config["meta_threshold"]
base_group = config["base_group"]
group_feature_names = config["group_feature_names"]

group_scalers = {}
specialists = {}

for group in group_feature_names.keys():
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{group}.joblib")
    model_path = os.path.join(MODEL_DIR, f"specialist_{group}.joblib")

    if os.path.exists(scaler_path):
        group_scalers[group] = joblib.load(scaler_path)

    if os.path.exists(model_path):
        specialists[group] = joblib.load(model_path)

meta_model = joblib.load(os.path.join(MODEL_DIR, "meta_model.joblib"))

print("=" * 60)
print("LOADED TRAINED ARTIFACTS")
print("=" * 60)
print("Groups:", list(group_feature_names.keys()))
print("Base group:", base_group)
print("Meta threshold:", meta_threshold)

# ============================================================
# PART 2: LOAD NEW CRANE DATA
# ============================================================
df_clean_with_lift = spark.table("default.df_raw_liftid_overload")
df_liftlog = spark.table("default.df_liftlog_overload")
df_liftlog = df_liftlog.filter(col("movementtype") != 5)

# Keep only lifts that survive the movementtype filter
valid_lift_ids = df_liftlog.select("lift_id").distinct()

df_clean_with_lift = (
    spark.table("default.df_raw_liftid_overload")
        .join(valid_lift_ids, on="lift_id", how="inner")
)
# ============================================================
# PART 5: DEFINE SAME GROUPS AS TRAINING
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
# PART 6: PREPROCESS NEW DATA
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
# PART 7: APPLY SAME FILLING LOGIC
# ============================================================

def weighted_fill_series(values, max_gap=10, zero_gap_fill=True):
    filled = values.astype(float).copy()
    n = len(filled)

    i = 0
    while i < n:
        if not np.isnan(filled[i]):
            i += 1
            continue

        start = i
        while i < n and np.isnan(filled[i]):
            i += 1
        end = i - 1

        left_idx = start - 1
        right_idx = end + 1

        left_valid = left_idx >= 0 and not np.isnan(filled[left_idx])
        right_valid = right_idx < n and not np.isnan(filled[right_idx])

        gap_length = end - start + 1

        if (
            zero_gap_fill
            and left_valid
            and right_valid
            and filled[left_idx] == 0
            and filled[right_idx] == 0
        ):
            filled[start:end + 1] = 0
            continue

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
    return index_series.astype(str).str.extract(r"^(\d+)_")[0]


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

        for col in data_cols:
            subdf[col] = weighted_fill_series(
                subdf[col].values,
                max_gap=max_gap,
                zero_gap_fill=(col in zero_fill_columns)
            )

        filled_parts.append(subdf)

    pdf_filled = pd.concat(filled_parts).sort_values(["_lift_id", "bucket_start_unix"])
    pdf_filled = pdf_filled.drop(columns=["_lift_id"])

    df_filled_dict_new[group] = pdf_filled

# ============================================================
# PART 8: SCORE NEW DATA WITH TRAINED MODELS
# ============================================================

def compute_row_activity(pdf, threshold=0.3):
    if pdf.shape[1] == 0:
        return pd.Series(False, index=pdf.index)
    row_max = np.abs(pdf.values).max(axis=1)
    return pd.Series(row_max > threshold, index=pdf.index)

def parse_lift_group_bucket(index_val):
    parts = str(index_val).split("_")
    return parts[0], int(parts[-1])

specialist_scores_new = {}
specialist_with_lift_new = {}

for group, pdf in df_filled_dict_new.items():
    if group not in group_scalers or group not in specialists:
        print(f"Skipping group '{group}' because model/scaler not found.")
        continue

    print(f"\nScoring new data group: {group}")

    pdf = pdf.copy()

    if "bucket_start_unix" not in pdf.columns:
        raise ValueError(f"Group '{group}' missing bucket_start_unix")

    feature_df = pdf.drop(columns=["bucket_start_unix"], errors="ignore").copy()
    feature_df.columns = [c.replace("ImportS7Device.", "") for c in feature_df.columns]
    feature_df = feature_df.select_dtypes(include=[np.number])

    expected_cols = group_feature_names[group]

    # Add missing expected columns
    for c in expected_cols:
        if c not in feature_df.columns:
            feature_df[c] = np.nan

    # Keep exactly the same column order as training
    feature_df = feature_df[expected_cols]

    # Final fill for ML compatibility
    feature_df = feature_df.fillna(feature_df.median())

    scaler = group_scalers[group]
    model = specialists[group]

    # Scale using training scaler
    feature_scaled = pd.DataFrame(
        scaler.transform(feature_df),
        index=feature_df.index,
        columns=feature_df.columns
    )

    # Activity mask
    activity_mask = compute_row_activity(
        feature_scaled,
        threshold=activity_thresholds.get(group, 0.3)
    )

    # Score with trained specialist
    raw_scores = model.decision_function(feature_scaled.values)

    # Use saved threshold from training if available
    threshold = specialist_thresholds.get(group, None)
    if threshold is None:
        # fallback if missing in config
        finite_scores = raw_scores[np.isfinite(raw_scores)]
        if len(finite_scores) > 0:
            q1 = np.percentile(finite_scores, 25)
            q3 = np.percentile(finite_scores, 75)
            iqr = q3 - q1
            threshold = q1 - 3.0 * iqr
        else:
            threshold = -np.inf

    # Predict anomaly only on active rows
    pred = np.where((activity_mask.values) & (raw_scores < threshold), -1, 1)

    scored_df = pd.DataFrame({
        f"{group}_score": raw_scores,
        f"{group}_pred": pred,
        f"{group}_active": activity_mask.astype(int).values,
        "bucket_start_unix": pdf["bucket_start_unix"].values
    }, index=pdf.index)

    specialist_scores_new[group] = scored_df

    # Add lift_id and timestamp
    parsed = [parse_lift_group_bucket(idx) for idx in scored_df.index]
    temp = scored_df.copy()
    temp["lift_id"] = [p[0] for p in parsed]
    temp["bucket_unix"] = [p[1] for p in parsed]
    temp["timestamp"] = pd.to_datetime(temp["bucket_unix"], unit="s", utc=True)

    specialist_with_lift_new[group] = temp.reset_index().rename(
        columns={"index": "lift_group_bucket"}
    )

    n_anom = (pred == -1).sum()
    n_active = activity_mask.sum()
    print(f"  Active rows: {n_active} / {len(activity_mask)}")
    print(f"  Threshold used: {threshold:.6f}")
    print(f"  Anomalies found: {n_anom} / {len(pred)}")

# ============================================================
# PART 9: ALIGN SPECIALISTS ONTO COMMON BASE TIMELINE
# ============================================================

if base_group not in specialist_with_lift_new:
    raise ValueError(
        f"Base group '{base_group}' not available in new scored data. "
        f"Available: {list(specialist_with_lift_new.keys())}"
    )

meta_features_new = specialist_with_lift_new[base_group][[
    "lift_group_bucket",
    "lift_id",
    "timestamp",
    f"{base_group}_score",
    f"{base_group}_pred",
    f"{base_group}_active"
]].copy()

meta_features_new = meta_features_new.sort_values(["lift_id", "timestamp"]).reset_index(drop=True)

for group, scores_df in specialist_with_lift_new.items():
    if group == base_group:
        continue

    print(f"Merging specialist group into meta timeline: {group}")

    right = scores_df[[
        "lift_id",
        "timestamp",
        f"{group}_score",
        f"{group}_pred",
        f"{group}_active"
    ]].sort_values(["lift_id", "timestamp"]).reset_index(drop=True)

    merged_parts = []

    for lid in meta_features_new["lift_id"].unique():
        left_lift = meta_features_new[meta_features_new["lift_id"] == lid].copy()
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
            merged = merged.drop(columns=[c for c in ["lift_id_x", "lift_id_y"] if c in merged.columns])

        merged_parts.append(merged)

    meta_features_new = pd.concat(merged_parts, ignore_index=True)
    print(f"  Shape after merge: {meta_features_new.shape}")

# Active specialist count
active_cols = [c for c in meta_features_new.columns if c.endswith("_active")]
meta_features_new["n_active_groups"] = meta_features_new[active_cols].fillna(0).sum(axis=1)

# Keep rows with at least one active subsystem
meta_features_new = meta_features_new[meta_features_new["n_active_groups"] > 0].copy()
meta_features_new = meta_features_new.reset_index(drop=True)

print("\nFinal meta feature table shape:", meta_features_new.shape)
print("Unique lifts:", meta_features_new["lift_id"].nunique())

# ============================================================
# PART 10: SCORE META MODEL
# ============================================================

score_columns = [c for c in meta_features_new.columns if c.endswith("_score")]
pred_columns = [c for c in meta_features_new.columns if c.endswith("_pred")]
meta_input_cols = score_columns + pred_columns

X_meta_df_new = meta_features_new[meta_input_cols].copy()
X_meta_df_new = X_meta_df_new.fillna(X_meta_df_new.median())

X_meta_new = X_meta_df_new.values

meta_features_new["meta_score"] = meta_model.decision_function(X_meta_new)
meta_features_new["meta_pred"] = np.where(
    meta_features_new["meta_score"] < meta_threshold,
    -1,
    1
)

n_meta_anom = (meta_features_new["meta_pred"] == -1).sum()
print("\n" + "=" * 60)
print("META MODEL RESULTS ON STRESS-TEST DATA")
print("=" * 60)
print(f"Meta anomalies found: {n_meta_anom} / {len(meta_features_new)} "
      f"({100 * n_meta_anom / len(meta_features_new):.2f}%)")

# ============================================================
# PART 11: SUMMARY PER LIFT
# ============================================================

anomaly_summary_new = (
    meta_features_new
    .groupby("lift_id")
    .agg(
        total_samples=("meta_pred", "count"),
        anomalies=("meta_pred", lambda x: (x == -1).sum()),
        worst_score=("meta_score", "min"),
        mean_score=("meta_score", "mean"),
        active_groups_mean=("n_active_groups", "mean")
    )
)

anomaly_summary_new["anomaly_pct"] = (
    100 * anomaly_summary_new["anomalies"] / anomaly_summary_new["total_samples"]
)

anomaly_summary_new = anomaly_summary_new.sort_values(
    ["anomalies", "worst_score"],
    ascending=[False, True]
)

print("\n" + "=" * 60)
print("ANOMALY SUMMARY PER LIFT - STRESS TEST")
print("=" * 60)
print(anomaly_summary_new.to_string())

# ============================================================
# PART 12: SPECIALIST ANOMALY RATES
# ============================================================

print("\n" + "=" * 60)
print("SPECIALIST ANOMALY RATES ON STRESS-TEST DATA")
print("=" * 60)

specialist_summary = []

for group, df_scores in specialist_scores_new.items():
    pred_col = f"{group}_pred"
    active_col = f"{group}_active"

    total_rows = len(df_scores)
    active_rows = int(df_scores[active_col].sum())
    anomalies = int((df_scores[pred_col] == -1).sum())

    anomaly_rate_all = 100 * anomalies / total_rows if total_rows > 0 else 0
    anomaly_rate_active = 100 * anomalies / active_rows if active_rows > 0 else 0

    specialist_summary.append({
        "group": group,
        "total_rows": total_rows,
        "active_rows": active_rows,
        "anomalies": anomalies,
        "anomaly_rate_all_pct": anomaly_rate_all,
        "anomaly_rate_active_pct": anomaly_rate_active
    })

specialist_summary_df = pd.DataFrame(specialist_summary).sort_values(
    "anomaly_rate_active_pct", ascending=False
)

print(specialist_summary_df.to_string(index=False))

# ============================================================
# PART 13: DID IT DETECT THE STRESS TEST AS ANOMALOUS?
# ============================================================

print("\n" + "=" * 60)
print("STRESS TEST DETECTION CHECK")
print("=" * 60)

if len(meta_features_new) == 0:
    print("No meta rows available after preprocessing/scoring.")
else:
    any_anomalies = (meta_features_new["meta_pred"] == -1).any()
    lifts_with_anomalies = (anomaly_summary_new["anomalies"] > 0).sum()

    print(f"Any anomalies detected? {'YES' if any_anomalies else 'NO'}")
    print(f"Lifts with at least one anomaly: {lifts_with_anomalies} / {len(anomaly_summary_new)}")

    if any_anomalies:
        print("The trained model detects anomalous behavior in the stress-test dataset.")
    else:
        print("The trained model did NOT detect anomalies in the stress-test dataset.")

# ============================================================
# PART 14: TOP ANOMALOUS WINDOWS
# ============================================================

top_anomalies = (
    meta_features_new.loc[meta_features_new["meta_pred"] == -1, [
        "lift_id", "timestamp", "meta_score", "n_active_groups"
    ]]
    .sort_values("meta_score", ascending=True)
    .head(50)
)

print("\n" + "=" * 60)
print("TOP ANOMALOUS TIME WINDOWS")
print("=" * 60)
print(top_anomalies.to_string(index=False))

# ============================================================
# PART 15: OPTIONAL MOVEMENT TYPE JOIN
# ============================================================

if "movementtype" in df_liftlog.columns:
    liftlog_pd = df_liftlog.select("lift_id", "movementtype").distinct().toPandas()
    liftid_to_movementtype = dict(zip(liftlog_pd["lift_id"].astype(str), liftlog_pd["movementtype"]))

    anomaly_summary_new = anomaly_summary_new.copy()
    anomaly_summary_new.index = anomaly_summary_new.index.astype(str)
    anomaly_summary_new["movementtype"] = anomaly_summary_new.index.map(liftid_to_movementtype.get)

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

        print(f"\nDriver summary for worst lift: {worst_lift}")

        for col_name in score_columns:
            pred_col = col_name.replace("_score", "_pred")

            if pred_col in lift_data.columns:
                print(
                    f"{col_name:35s} "
                    f"anom_windows={(lift_data[pred_col] == -1).sum():5d} "
                    f"mean_score={lift_data[col_name].mean(): .4f} "
                    f"min_score={lift_data[col_name].min(): .4f}"
                )
    anomaly_mask = lift_data["meta_pred"] == -1

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
        ax.plot(lift_data["timestamp"], lift_data[col_name], linewidth=0.8, alpha=0.85)
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
        linewidth=1.0,
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
    ax.set_ylabel("META", fontsize=10, fontweight="bold")

    plt.suptitle(f"Stress-Test Crane: Lift {worst_lift} - Anomaly Scores", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print(f"WHAT DRIVES ANOMALIES ON STRESS-TEST LIFT {worst_lift}?")
    print("=" * 60)

    n_normal = (~anomaly_mask).sum()
    n_anom = anomaly_mask.sum()

    print(f"Normal windows: {n_normal}")
    print(f"Anomalous windows: {n_anom}")

    for col_name in score_columns:
        normal_vals = lift_data.loc[~anomaly_mask, col_name]
        anomaly_vals = lift_data.loc[anomaly_mask, col_name]

        normal_mean = normal_vals.mean() if len(normal_vals) > 0 else None
        anomaly_mean = anomaly_vals.mean() if len(anomaly_vals) > 0 else None

        normal_str = f"{normal_mean:.4f}" if normal_mean is not None else "no normal rows"
        anomaly_str = f"{anomaly_mean:.4f}" if anomaly_mean is not None else "no anomaly rows"

        print(f"{col_name:40s} normal={normal_str} anomaly={anomaly_str}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col

df_alarms = spark.table("default.df_alarms_fit_overload")

most_anom_lift = str(anomaly_summary_new.index[0])

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

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col

df_alarms = spark.table("default.df_alarms_fit_overload")

lift_ids = ["1184", "1233", "1203", "1207", "1349", "1302", "1331", "1304"]
display(
    df_alarms
    .filter(col("lift_id").cast("string").isin(lift_ids))
    .select(
        "lift_id",
        "timestamp_utc",
        "id",
        "text",
        "status",
        "onstart",
        "distance_seconds",
        "id_fitted",
        "cause",
        "corrective_action"
    )
    .orderBy("lift_id", "timestamp_utc")
)
display(
    df_alarms
    .filter(col("lift_id").cast("string").isin(lift_ids))
    .groupBy("lift_id", "id", "text")
    .agg(
        F.count("*").alias("alarm_count"),
        F.min("timestamp_utc").alias("first_alarm"),
        F.max("timestamp_utc").alias("last_alarm"),
        F.min("distance_seconds").alias("min_distance_seconds"),
        F.max("distance_seconds").alias("max_distance_seconds")
    )
    .orderBy("lift_id", F.desc("alarm_count"))
)
