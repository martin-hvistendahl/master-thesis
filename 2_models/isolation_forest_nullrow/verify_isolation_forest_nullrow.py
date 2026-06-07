# Databricks notebook source
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

MODEL_DIR = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/anomaly_model_null_rows"
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
score_columns_train = config.get("score_columns", [])
pred_columns_train = config.get("pred_columns", [])

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
print("LOADED NULL-ROW TRAINED ARTIFACTS")
print("=" * 60)
print("Groups:", list(group_feature_names.keys()))
print("Base group:", base_group)
print("Meta threshold:", meta_threshold)

# ============================================================
# HELPERS
# ============================================================

def compute_row_activity(pdf, threshold=0.3):
    if pdf.shape[1] == 0:
        return pd.Series(False, index=pdf.index)
    row_max = np.abs(pdf.values).max(axis=1)
    return pd.Series(row_max > threshold, index=pdf.index)

def parse_lift_group_bucket(index_val):
    parts = str(index_val).split("_")
    return parts[0], int(parts[-1])

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
    if group not in group_feature_names:
        continue

    df_group = df_raw_buckets.filter(F.col("tag_group") == group)

    if df_group.limit(1).count() > 0:
        df_pivoted_dict_new[group] = (
            df_group
            .groupBy("lift_group_bucket", "bucket_start_unix")
            .pivot("tagname", tags)
            .agg(F.mean("value"))
            .orderBy("lift_group_bucket")
        )

print("\nAvailable pivoted groups in new data:")
for g in df_pivoted_dict_new:
    print(" -", g)

# ============================================================
# PART 7: KEEP NULLS, THEN REMOVE NULL ROWS BEFORE SCORING
# ============================================================

df_null_dict_new = {}

for group, spark_df in df_pivoted_dict_new.items():
    print(f"\nPreparing NULL-row verification group: {group}")

    pdf = spark_df.toPandas()
    pdf = pdf.sort_values(["lift_group_bucket", "bucket_start_unix"]).copy()
    pdf = pdf.set_index("lift_group_bucket", drop=True)

    null_pct = pdf.drop(columns=["bucket_start_unix"], errors="ignore").isna().mean().mean()
    print(f"  Mean NULL percentage before row drop: {100 * null_pct:.2f}%")

    df_null_dict_new[group] = pdf

# ============================================================
# PART 8: SCORE NEW DATA WITH TRAINED SPECIALISTS
# ============================================================

specialist_scores_new = {}
specialist_with_lift_new = {}

for group, pdf in df_null_dict_new.items():
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

    for c in expected_cols:
        if c not in feature_df.columns:
            feature_df[c] = np.nan

    feature_df = feature_df[expected_cols]

    before_rows = len(feature_df)
    valid_mask = ~feature_df.isna().any(axis=1)
    feature_df = feature_df.loc[valid_mask].copy()
    pdf_valid = pdf.loc[valid_mask].copy()
    after_rows = len(feature_df)

    print(f"  Removed NULL rows: {before_rows - after_rows} / {before_rows}")
    print(f"  Remaining complete rows: {after_rows}")

    if after_rows == 0:
        print(f"  Skipping group '{group}' because no complete rows remain.")
        continue

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

    raw_scores = model.decision_function(feature_scaled.values)

    threshold = specialist_thresholds.get(group, None)

    if threshold is None:
        finite_scores = raw_scores[np.isfinite(raw_scores)]
        threshold = np.percentile(finite_scores, 1) if len(finite_scores) > 0 else -np.inf

    pred = np.where((activity_mask.values) & (raw_scores < threshold), -1, 1)

    scored_df = pd.DataFrame({
        f"{group}_score": raw_scores,
        f"{group}_pred": pred,
        f"{group}_active": activity_mask.astype(int).values,
        "bucket_start_unix": pdf_valid["bucket_start_unix"].values
    }, index=feature_scaled.index)

    specialist_scores_new[group] = scored_df

    parsed = [parse_lift_group_bucket(idx) for idx in scored_df.index]
    temp = scored_df.copy()
    temp["lift_id"] = [p[0] for p in parsed]
    temp["bucket_unix"] = [p[1] for p in parsed]
    temp["timestamp"] = pd.to_datetime(temp["bucket_unix"], unit="s", utc=True)

    specialist_with_lift_new[group] = (
        temp.reset_index()
        .rename(columns={"index": "lift_group_bucket"})
        .sort_values(["lift_id", "timestamp"])
        .reset_index(drop=True)
    )

    n_anom = int((pred == -1).sum())
    n_active = int(activity_mask.sum())

    print(f"  Active rows: {n_active} / {len(activity_mask)}")
    print(f"  Threshold used: {float(threshold):.6f}")
    print(f"  Specialist anomalies found: {n_anom} / {len(pred)}")

if len(specialist_with_lift_new) == 0:
    raise ValueError("No groups could be scored after removing NULL rows.")

# ============================================================
# PART 9: ALIGN SPECIALISTS ONTO COMMON BASE TIMELINE
# ============================================================

if base_group not in specialist_with_lift_new:
    available_groups = list(specialist_with_lift_new.keys())
    fallback_base_group = min(available_groups, key=lambda g: bucket_sizes.get(g, 999999))
    print(
        f"\nBase group '{base_group}' unavailable after NULL removal. "
        f"Using fallback base group '{fallback_base_group}'."
    )
    base_group_new = fallback_base_group
else:
    base_group_new = base_group

meta_features_new = specialist_with_lift_new[base_group_new][[
    "lift_group_bucket",
    "lift_id",
    "timestamp",
    f"{base_group_new}_score",
    f"{base_group_new}_pred",
    f"{base_group_new}_active"
]].copy()

meta_features_new = meta_features_new.sort_values(["lift_id", "timestamp"]).reset_index(drop=True)

for group, scores_df in specialist_with_lift_new.items():
    if group == base_group_new:
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

active_cols = [c for c in meta_features_new.columns if c.endswith("_active")]
meta_features_new["n_active_groups"] = meta_features_new[active_cols].fillna(0).sum(axis=1)

meta_features_new = meta_features_new[meta_features_new["n_active_groups"] > 0].copy()
meta_features_new = meta_features_new.reset_index(drop=True)

print("\nFinal meta feature table shape:", meta_features_new.shape)
print("Unique lifts:", meta_features_new["lift_id"].nunique() if len(meta_features_new) else 0)

if len(meta_features_new) == 0:
    raise ValueError("No active meta rows remain after NULL-row filtering.")

# ============================================================
# PART 10: SCORE META MODEL
# ============================================================

for col_name in score_columns_train:
    if col_name not in meta_features_new.columns:
        meta_features_new[col_name] = np.nan

for col_name in pred_columns_train:
    if col_name not in meta_features_new.columns:
        meta_features_new[col_name] = np.nan

meta_input_cols = score_columns_train + pred_columns_train

if len(meta_input_cols) == 0:
    score_columns_train = [c for c in meta_features_new.columns if c.endswith("_score")]
    pred_columns_train = [c for c in meta_features_new.columns if c.endswith("_pred")]
    meta_input_cols = score_columns_train + pred_columns_train

X_meta_df_new = meta_features_new[meta_input_cols].copy()

# This fills only missing specialist outputs after alignment,
# not raw sensor values.
X_meta_df_new = X_meta_df_new.fillna(X_meta_df_new.median(numeric_only=True))
X_meta_df_new = X_meta_df_new.fillna(0)

X_meta_new = X_meta_df_new.values

meta_features_new["meta_score"] = meta_model.decision_function(X_meta_new)
meta_features_new["meta_pred"] = np.where(
    (meta_features_new["n_active_groups"] > 0) &
    (meta_features_new["meta_score"] < meta_threshold),
    -1,
    1
)

n_meta_anom = int((meta_features_new["meta_pred"] == -1).sum())

print("\n" + "=" * 60)
print("META MODEL RESULTS ON NULL-ROW VERIFICATION DATA")
print("=" * 60)
print(
    f"Meta anomalies found: {n_meta_anom} / {len(meta_features_new)} "
    f"({100 * n_meta_anom / len(meta_features_new):.2f}%)"
)

# ============================================================
# PART 11: SUMMARY PER LIFT
# ============================================================

anomaly_summary_new = (
    meta_features_new
    .groupby("lift_id")
    .agg(
        total_samples=("meta_pred", "count"),
        anomalies=("meta_pred", lambda x: int((x == -1).sum())),
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
print("ANOMALY SUMMARY PER LIFT - NULL-ROW VERIFICATION")
print("=" * 60)
print(anomaly_summary_new.to_string())

# ============================================================
# PART 12: SPECIALIST ANOMALY RATES
# ============================================================

print("\n" + "=" * 60)
print("SPECIALIST ANOMALY RATES")
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
        "total_rows_after_null_drop": total_rows,
        "active_rows": active_rows,
        "anomalies": anomalies,
        "anomaly_rate_all_pct": anomaly_rate_all,
        "anomaly_rate_active_pct": anomaly_rate_active
    })

specialist_summary_df = pd.DataFrame(specialist_summary).sort_values(
    "anomaly_rate_active_pct",
    ascending=False
)

print(specialist_summary_df.to_string(index=False))

# ============================================================
# PART 13: DID IT DETECT ANOMALIES?
# ============================================================

print("\n" + "=" * 60)
print("ANOMALY DETECTION CHECK")
print("=" * 60)

any_anomalies = (meta_features_new["meta_pred"] == -1).any()
lifts_with_anomalies = int((anomaly_summary_new["anomalies"] > 0).sum())

print(f"Any anomalies detected? {'YES' if any_anomalies else 'NO'}")
print(f"Lifts with at least one anomaly: {lifts_with_anomalies} / {len(anomaly_summary_new)}")

if any_anomalies:
    print("The NULL-row model detects anomalous behavior in this verification dataset.")
else:
    print("The NULL-row model did NOT detect anomalies in this verification dataset.")

# ============================================================
# PART 14: TOP ANOMALOUS WINDOWS
# ============================================================

top_anomalies = (
    meta_features_new.loc[
        meta_features_new["meta_pred"] == -1,
        ["lift_id", "timestamp", "meta_score", "n_active_groups"]
    ]
    .sort_values("meta_score", ascending=True)
    .head(50)
)

print("\n" + "=" * 60)
print("TOP ANOMALOUS TIME WINDOWS")
print("=" * 60)
print(top_anomalies.to_string(index=False))

# ============================================================
# PART 15: MOVEMENT TYPE JOIN
# ============================================================

if "movementtype" in df_liftlog.columns:
    liftlog_pd = df_liftlog.select("lift_id", "movementtype").distinct().toPandas()
    liftid_to_movementtype = dict(
        zip(liftlog_pd["lift_id"].astype(str), liftlog_pd["movementtype"])
    )

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

score_columns_plot = [c for c in meta_features_new.columns if c.endswith("_score")]

if len(anomaly_summary_new) > 0:
    worst_lift = anomaly_summary_new.index[0]

    lift_data = meta_features_new[
        meta_features_new["lift_id"].astype(str) == str(worst_lift)
    ].copy().sort_values("timestamp")

    anomaly_mask = lift_data["meta_pred"] == -1

    fig, axes = plt.subplots(
        len(score_columns_plot) + 1,
        1,
        figsize=(20, 3 * (len(score_columns_plot) + 1)),
        sharex=True
    )

    if len(score_columns_plot) == 0:
        axes = [axes]

    for i, col_name in enumerate(score_columns_plot):
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

    plt.suptitle(f"NULL-row Verification: Lift {worst_lift} - Anomaly Scores", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print(f"WHAT DRIVES ANOMALIES ON LIFT {worst_lift}?")
    print("=" * 60)

    n_normal = int((~anomaly_mask).sum())
    n_anom = int(anomaly_mask.sum())

    print(f"Normal windows: {n_normal}")
    print(f"Anomalous windows: {n_anom}")

    for col_name in score_columns_plot:
        normal_vals = lift_data.loc[~anomaly_mask, col_name]
        anomaly_vals = lift_data.loc[anomaly_mask, col_name]

        normal_mean = normal_vals.mean() if len(normal_vals) > 0 else None
        anomaly_mean = anomaly_vals.mean() if len(anomaly_vals) > 0 else None

        normal_str = f"{normal_mean:.4f}" if normal_mean is not None else "no normal rows"
        anomaly_str = f"{anomaly_mean:.4f}" if anomaly_mean is not None else "no anomaly rows"

        print(f"{col_name:40s} normal={normal_str} anomaly={anomaly_str}")
else:
    print("No lifts available for plotting.")

# COMMAND ----------

