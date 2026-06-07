# Databricks notebook source
# Databricks notebook source
# ============================================================
# Verification script for saved per-tag Isolation Forest models
# ============================================================

import os
import joblib
import gc
import random
import numpy as np
import pandas as pd

from scipy import stats, signal
from numpy.fft import rfft

from pyspark.sql import functions as F
from pyspark.sql import types as T

# ============================================================
# 1. Reproducibility and paths
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_DIR = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/singel_tag_anomali"

MODEL_PATH = os.path.join(MODEL_DIR, "per_tag_iforest_models.joblib")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")
TAGS_PATH = os.path.join(MODEL_DIR, "tags_of_interest.joblib")

# ============================================================
# 2. Load trained models
# ============================================================

per_tag_models = joblib.load(MODEL_PATH)
FEATURE_NAMES = joblib.load(FEATURE_PATH)
tags_of_interest = joblib.load(TAGS_PATH)

trained_tags = sorted(list(per_tag_models.keys()))

print(f"Loaded models for {len(trained_tags)} tags")
print(f"Loaded {len(FEATURE_NAMES)} feature names")

# ============================================================
# 3. Load verification data
# ============================================================

from pyspark.sql.functions import col
df_liftlog_verify = spark.table("default.df_liftlog_overload")
df_liftlog_verify = df_liftlog_verify.filter(col("movementtype") != 5)

valid_lift_ids = df_liftlog_verify.select("lift_id").distinct()

df_raw_verify = (
    spark.table("default.df_raw_liftid_overload")
        .join(valid_lift_ids, on="lift_id", how="inner")
)

n_input_lifts = df_raw_verify.select("lift_id").distinct().count()
print(f"\nLifts entering pipeline (after movementtype filter): {n_input_lifts}")

# ============================================================
# 4. Same feature extraction as training
# ============================================================

MIN_SAMPLES_PER_LIFT = 20

def _safe(x, default=0.0):
    return default if (x is None or not np.isfinite(x)) else float(x)

def _spectral_entropy(psd):
    psd = psd[psd > 0]
    if psd.size == 0:
        return 0.0
    p = psd / psd.sum()
    return float(-(p * np.log(p)).sum())

def extract_features(values: np.ndarray, dur_sec: float) -> np.ndarray:
    v = values[~np.isnan(values)].astype(float)
    n = len(v)

    if n < 5 or dur_sec <= 0:
        return np.full(len(FEATURE_NAMES), np.nan)

    mean_ = np.mean(v)
    std_ = np.std(v)
    median_ = np.median(v)

    q25, q75 = np.percentile(v, [25, 75])
    iqr_ = q75 - q25

    sk_ = _safe(stats.skew(v))
    ku_ = _safe(stats.kurtosis(v))

    min_ = np.min(v)
    max_ = np.max(v)
    ptp_ = max_ - min_

    p05_ = np.percentile(v, 5)
    p95_ = np.percentile(v, 95)

    energy_per_sec = np.trapz(v * v) / dur_sec
    abs_mean_diff_per_sec = (
        np.mean(np.abs(np.diff(v))) * (n / dur_sec)
        if n > 1 else 0.0
    )

    peaks, props = signal.find_peaks(v)
    n_peaks_per_sec = len(peaks) / dur_sec
    mean_peak_height = float(np.mean(v[peaks])) if len(peaks) > 0 else 0.0

    net_slope_per_sec = (v[-1] - v[0]) / dur_sec

    v_centered = v - mean_
    zero_cross = np.sum(np.diff(np.signbit(v_centered)))
    zero_cross_per_sec = zero_cross / dur_sec

    v_detrend = v - mean_
    fft_mag = np.abs(rfft(v_detrend))

    if fft_mag.sum() > 0:
        freqs = np.arange(len(fft_mag))
        p = fft_mag / (fft_mag.sum() + 1e-12)

        spectral_centroid = float(np.sum(freqs * p))
        spectral_bandwidth = float(
            np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * p))
        )
        spectral_entropy_ = _spectral_entropy(fft_mag ** 2)

        top3 = np.sort(fft_mag)[-3:][::-1] / (fft_mag.sum() + 1e-12)
        fft_top1_rel = float(top3[0]) if len(top3) > 0 else 0.0
        fft_top2_rel = float(top3[1]) if len(top3) > 1 else 0.0
        fft_top3_rel = float(top3[2]) if len(top3) > 2 else 0.0
    else:
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0
        spectral_entropy_ = 0.0
        fft_top1_rel = 0.0
        fft_top2_rel = 0.0
        fft_top3_rel = 0.0

    def _autocorr(x, lag):
        if len(x) <= lag or np.std(x) == 0:
            return 0.0
        c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        return _safe(c)

    ac1 = _autocorr(v, 1)
    ac5 = _autocorr(v, 5)

    return np.array([
        mean_, std_, median_, iqr_,
        sk_, ku_,
        min_, max_, ptp_, p05_, p95_,
        energy_per_sec, abs_mean_diff_per_sec,
        n_peaks_per_sec, mean_peak_height,
        net_slope_per_sec, zero_cross_per_sec,
        spectral_centroid, spectral_bandwidth, spectral_entropy_,
        fft_top1_rel, fft_top2_rel, fft_top3_rel,
        ac1, ac5,
        float(n),
    ])

feature_schema = T.StructType(
    [T.StructField("lift_id", T.StringType(), False),
     T.StructField("tagname", T.StringType(), False)] +
    [T.StructField(c, T.DoubleType(), True) for c in FEATURE_NAMES]
)

def feature_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values("ts")

    lift_id = str(pdf["lift_id"].iloc[0])
    tagname = str(pdf["tagname"].iloc[0])

    if len(pdf) < MIN_SAMPLES_PER_LIFT:
        return pd.DataFrame([], columns=["lift_id", "tagname"] + FEATURE_NAMES)

    dur_sec = (pdf["ts"].max() - pdf["ts"].min()).total_seconds()

    if dur_sec <= 0:
        return pd.DataFrame([], columns=["lift_id", "tagname"] + FEATURE_NAMES)

    feats = extract_features(pdf["value"].to_numpy(dtype=float), dur_sec)

    return pd.DataFrame(
        [[lift_id, tagname] + feats.tolist()],
        columns=["lift_id", "tagname"] + FEATURE_NAMES
    )

# ============================================================
# 5. Prepare verification time-series data
# ============================================================

df_ts_verify = (
    df_raw_verify
    .filter(F.col("tagname").isin(trained_tags))
    .withColumn("value", F.col("value").cast("double"))
    .filter(F.col("value").isNotNull())
    .withColumn("ts", F.col("timestamp_utc").cast("timestamp"))
    .select("lift_id", "tagname", "ts", "value")
    .repartition("tagname")
    .cache()
)

_ = df_ts_verify.count()

# ============================================================
# 6. Generate verification features
# ============================================================

df_features_verify = (
    df_ts_verify
    .repartition("tagname", "lift_id")
    .groupBy("tagname", "lift_id")
    .applyInPandas(feature_udf, schema=feature_schema)
    .cache()
)

n_feature_rows = df_features_verify.count()
print(f"Feature rows created: {n_feature_rows}")

# ============================================================
# 7. Score verification data with saved models
# ============================================================

verification_results = []

for tag, pipe in per_tag_models.items():
    try:
        pdf = (
            df_features_verify
            .filter(F.col("tagname") == tag)
            .toPandas()
        )

        if pdf.empty:
            print(f"[SKIP] {tag}: no verification data")
            continue

        for c in FEATURE_NAMES:
            if c not in pdf.columns:
                pdf[c] = np.nan

        lift_ids = pdf["lift_id"].to_numpy()
        X = pdf[FEATURE_NAMES].to_numpy(dtype=float)

        imputer = pipe.named_steps["imputer"]
        scaler = pipe.named_steps["scaler"]
        iforest = pipe.named_steps["iforest"]

        n_expected = imputer.statistics_.shape[0]

        valid_cols = ~np.all(np.isnan(X), axis=0)
        X_valid = X[:, valid_cols]

        if X_valid.shape[1] != n_expected:
            print(
                f"[SKIP] {tag}: feature mismatch "
                f"{X_valid.shape[1]} vs expected {n_expected}"
            )
            continue

        X_imp = imputer.transform(X_valid)
        X_sc = scaler.transform(X_imp)

        raw_scores = iforest.score_samples(X_sc)

        anomaly_score = -raw_scores
        is_anomaly = (iforest.predict(X_sc) == -1).astype(int)

        verification_results.append(pd.DataFrame({
            "lift_id": lift_ids,
            "tagname": tag,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
        }))

        print(f"[OK] {tag}: {len(pdf)} lifts scored, {int(is_anomaly.sum())} anomalies")

    except Exception as e:
        print(f"[ERR] {tag}: {e}")
        continue

    gc.collect()

# ============================================================
# 8. Combine results
# ============================================================

if len(verification_results) == 0:
    results_verify_df = pd.DataFrame(
        columns=["lift_id", "tagname", "anomaly_score", "is_anomaly"]
    )
    print("No verification results produced.")
else:
    results_verify_df = pd.concat(verification_results, ignore_index=True)
    print(f"Verification result rows: {len(results_verify_df)}")

results_verify_sdf = spark.createDataFrame(results_verify_df)

# ============================================================
# 9. Join with liftlog metadata
# ============================================================

df_liftlog_verify_small = (
    df_liftlog_verify
    .withColumn("lift_id", F.col("lift_id").cast("string"))
)

results_verify_with_liftlog = (
    results_verify_sdf
    .withColumn("lift_id", F.col("lift_id").cast("string"))
    .join(df_liftlog_verify_small, on="lift_id", how="left")
)

# ============================================================
# 10. Lift-level verification summary
# ============================================================

lift_verification_summary = (
    results_verify_with_liftlog
    .groupBy("lift_id")
    .agg(
        F.count("*").alias("n_tag_scores"),
        F.sum("is_anomaly").alias("n_anomalous_tags"),
        F.mean("anomaly_score").alias("mean_anomaly_score"),
        F.max("anomaly_score").alias("max_anomaly_score")
    )
    .withColumn(
        "anomalous_tag_rate",
        F.col("n_anomalous_tags") / F.col("n_tag_scores")
    )
    .orderBy(F.desc("n_anomalous_tags"), F.desc("max_anomaly_score"))
)

# ============================================================
# 11. Save verification results
# ============================================================

results_verify_sdf.write.mode("overwrite").format("delta").saveAsTable(
    "default.verification_iforest_tag_scores_overload"
)

results_verify_with_liftlog.write.mode("overwrite").format("delta").saveAsTable(
    "default.verification_iforest_tag_scores_with_liftlog_overload"
)

lift_verification_summary.write.mode("overwrite").format("delta").saveAsTable(
    "default.verification_iforest_lift_summary_overload"
)

# ============================================================
# 12. Display results
# ============================================================

display(results_verify_with_liftlog.orderBy(F.desc("anomaly_score")))
display(lift_verification_summary)

# ============================================================
# 13. Clean cache
# ============================================================

df_ts_verify.unpersist()
df_features_verify.unpersist()

# COMMAND ----------

# ============================================================
# 12A. OVERALL VERIFICATION SUMMARY
# ============================================================

overall_summary = pd.DataFrame([{
    "trained_tags": len(trained_tags),
    "input_lifts_after_filter": n_input_lifts,
    "feature_rows_created": n_feature_rows,
    "verification_score_rows": len(results_verify_df),
    "unique_lifts_scored": results_verify_df["lift_id"].nunique(),
    "unique_tags_scored": results_verify_df["tagname"].nunique(),
    "total_anomalous_tag_scores": int(results_verify_df["is_anomaly"].sum()),
    "overall_anomalous_tag_rate_pct": 100 * results_verify_df["is_anomaly"].mean()
}])

display(overall_summary)

# ============================================================
# 12B. TAG-LEVEL SUMMARY
# ============================================================

tag_summary = (
    results_verify_df
    .groupby("tagname")
    .agg(
        lifts_scored=("lift_id", "count"),
        anomalous_lifts=("is_anomaly", "sum"),
        mean_anomaly_score=("anomaly_score", "mean"),
        max_anomaly_score=("anomaly_score", "max")
    )
    .reset_index()
)

tag_summary["anomalous_lift_rate_pct"] = (
    100 * tag_summary["anomalous_lifts"] / tag_summary["lifts_scored"]
)

tag_summary = tag_summary.sort_values(
    ["anomalous_lifts", "max_anomaly_score"],
    ascending=[False, False]
)

display(tag_summary)

# ============================================================
# 12C. TOP ANOMALOUS TAG-LIFT COMBINATIONS
# ============================================================

top_tag_lift_anomalies = (
    results_verify_df
    .sort_values("anomaly_score", ascending=False)
    .head(50)
)

display(top_tag_lift_anomalies)

# ============================================================
# 12D. IMPROVED LIFT-LEVEL SUMMARY
# ============================================================

lift_summary_pd = (
    results_verify_df
    .groupby("lift_id")
    .agg(
        n_tag_scores=("tagname", "count"),
        n_anomalous_tags=("is_anomaly", "sum"),
        mean_anomaly_score=("anomaly_score", "mean"),
        max_anomaly_score=("anomaly_score", "max")
    )
    .reset_index()
)

lift_summary_pd["anomalous_tag_rate_pct"] = (
    100 * lift_summary_pd["n_anomalous_tags"] / lift_summary_pd["n_tag_scores"]
)

lift_summary_pd = lift_summary_pd.sort_values(
    ["n_anomalous_tags", "max_anomaly_score"],
    ascending=[False, False]
)

display(lift_summary_pd)