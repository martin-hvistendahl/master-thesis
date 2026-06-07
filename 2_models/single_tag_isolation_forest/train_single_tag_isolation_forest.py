# Databricks notebook source
# ============================================
# 1. Imports
# ============================================
from pyspark.sql import functions as F
from pyspark.sql.functions import col, coalesce, when

import os
import glob
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.ensemble import IsolationForest
from pyspark.sql import functions as F



# COMMAND ----------



# ============================================
# 2. Reproducibility
# ============================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# ============================================
# 3. Configuration
# ============================================

df_raw_with_lift = spark.table("default.df_all_raw_liftid")
df_liftlog = spark.table("default.all_liftlog")


df_liftlog_5 = df_liftlog.filter(F.col("movementtype") == 5)
df_liftlog = df_liftlog.filter(F.col("movementtype") != 5)



SEQ_LEN = 128
MIN_LIFTS_PER_TAG = 30

# contamination only affects isolation-forest ranking threshold behavior
ISOF_CONTAMINATION = 0.02

tags_of_interest = [
    "ImportS7Device.Drives.MainRectifier.Power",
    "ImportS7Device.Drives.EmergRectifier.Power",
    "ImportS7Device.Drives.MainRectifier.DCLinkVoltage",
    "ImportS7Device.Drives.EmergRectifier.DCLinkVoltage",
    "ImportS7Device.Drives.MainRectifier.SupplyVoltage",
    "ImportS7Device.Drives.EmergRectifier.SupplyVoltage",
    "ImportS7Device.General.HoistLoadPctSWL",
    "ImportS7Device.General.BoomLoad",
    "ImportS7Device.General.PedestalMoment",
    "ImportS7Device.SafetySystems.AOPS RopeForce",
    "ImportS7Device.CBM.LuffingWinch_Moment",
    "ImportS7Device.CBM.MainHoistWinch_Moment",
    "ImportS7Device.CBM.SlewBearing_Moment",
    "ImportS7Device.General.SlewMoment",
    "ImportS7Device.CBM.SlewGear_Moment",
    "ImportS7Device.Drives.Hoist_Status.ActualTorque",
    "ImportS7Device.Drives.Hoist_Status.ActualCurrent",
    "ImportS7Device.Drives.Hoist_Status.ActualSpeed",
    "ImportS7Device.Drives.Hoist_Status.ActualPower",
    "ImportS7Device.Drives.Boom_Status.ActualTorque",
    "ImportS7Device.Drives.Boom_Status.ActualCurrent",
    "ImportS7Device.Drives.Boom_Status.ActualSpeed",
    "ImportS7Device.Drives.Boom_Status.ActualPower",
    "ImportS7Device.Drives.SlewA_Status.ActualTorque",
    "ImportS7Device.Drives.SlewB_Status.ActualTorque",
    "ImportS7Device.Drives.SlewA_Status.ActualSpeed",
    "ImportS7Device.Drives.SlewB_Status.ActualSpeed",
    "ImportS7Device.Drives.SlewA_Status.ActualCurrent",
    "ImportS7Device.Drives.SlewB_Status.ActualCurrent",
    "ImportS7Device.Drives.SlewA_Status.ActualPower",
    "ImportS7Device.Drives.SlewB_Status.ActualPower",
    "ImportS7Device.Drives.Hoist_Control.SpeedReference",
    "ImportS7Device.Drives.Boom_Control.SpeedReference",
    "ImportS7Device.Drives.SlewA_Control.SpeedReference",
    "ImportS7Device.Drives.SlewB_Control.SpeedReference",
    "ImportS7Device.Joysticks.Hoist",
    "ImportS7Device.Joysticks.Boom",
    "ImportS7Device.Joysticks.Slew",
    "ImportS7Device.Hyd.BrakeSystemPressure",
    "ImportS7Device.Hyd.BrakeAccumulatorPressure",
    "ImportS7Device.Hyd.BrakesEnablePressure",
    "ImportS7Device.Hyd.SlewBrakePressureA",
    "ImportS7Device.Hyd.SlewBrakePressureB",
    "ImportS7Device.Hyd.HoistSecondaryPressure",
    "ImportS7Device.Hyd.HoistPrimaryPressure",
    "ImportS7Device.Hyd.BoomSecondaryPressure",
    "ImportS7Device.Hyd.BoomPrimaryPressure",
    "ImportS7Device.General.HoistPosition",
    "ImportS7Device.Encoders.MainA_Raw",
    "ImportS7Device.Encoders.MainB_Raw",
    "ImportS7Device.Encoders.BoomA_Raw",
    "ImportS7Device.Encoders.BoomB_Raw",
    "ImportS7Device.Encoders.Slew_Raw",
    "ImportS7Device.General.BoomAngle",
    "ImportS7Device.General.BoomRadius",
    "ImportS7Device.General.SlewAngle",
    "ImportS7Device.General.HeelAngle",
    "ImportS7Device.General.TrimAngle",
    "ImportS7Device.General.WindSpeed",
    "ImportS7Device.Drives.Hoist_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.Boom_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.SlewA_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.SlewB_Status.DepletionLayerMaxTemp",
    "ImportS7Device.Drives.Hoist_Status.MotorTemp",
    "ImportS7Device.Drives.Boom_Status.MotorTemp",
    "ImportS7Device.Drives.SlewA_Status.MotorTemp",
    "ImportS7Device.Drives.SlewB_Status.MotorTemp",
    "ImportS7Device.HVAC.ResistorCoolant.ResistorTankCoolantTemp",
    "ImportS7Device.HVAC.ResistorCoolant.BrakeResistorTempA",
    "ImportS7Device.HVAC.ResistorCoolant.BrakeResistorTempB",
    "ImportS7Device.Drives.SlewA_Status.DroopFeedbackSpeedReduction",
    "ImportS7Device.Drives.SlewB_Status.DroopFeedbackSpeedReduction",
    "ImportS7Device.Drives.SlewA_Status.TorqueLimEffective",
    "ImportS7Device.Drives.SlewB_Status.TorqueLimEffective"
]

tags_of_interest = sorted(list(set(tags_of_interest)))
print("Unique tags:", len(tags_of_interest))

# COMMAND ----------

# ============================================
# 1. Imports
# ============================================
from pyspark.sql import functions as F

import numpy as np
import pandas as pd
from scipy import stats, signal
from numpy.fft import rfft

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================
# 2. Reproducibility & Config
# ============================================
SEED = 42
np.random.seed(SEED)

MIN_LIFTS_PER_TAG = 30
MIN_SAMPLES_PER_LIFT = 20      # skip lifts with too few raw points for a tag
ISOF_CONTAMINATION = 0.02
N_ESTIMATORS = 200

# Assumes df_raw_with_lift, df_liftlog, tags_of_interest already defined.

# ============================================
# 3. Prepare Spark dataframe (no toPandas here!)
# ============================================
df_ts = (
    df_raw_with_lift
    .filter(F.col("tagname").isin(tags_of_interest))
    .withColumn("value", F.col("value").cast("double"))
    .filter(F.col("value").isNotNull())
    .withColumn("ts", F.col("timestamp_utc").cast("timestamp"))
    .select("lift_id", "tagname", "ts", "value")
)

# Cache the filtered set once (tags_of_interest is small vs full raw table)
df_ts = df_ts.repartition("tagname").cache()
_ = df_ts.count()   # materialize cache

# Which tags actually have data?
available_tags = [r["tagname"] for r in
                  df_ts.select("tagname").distinct().collect()]
print(f"Tags available: {len(available_tags)}")


# COMMAND ----------

# ============================================
# 4. Duration-invariant feature extractor
# ============================================
FEATURE_NAMES = [
    # Central tendency / dispersion
    "mean", "std", "median", "iqr",
    # Shape of the distribution
    "skew", "kurtosis",
    # Range / extremes
    "min", "max", "ptp", "p05", "p95",
    # Rate-normalized (per-second) dynamics
    "energy_per_sec", "abs_mean_diff_per_sec",
    "n_peaks_per_sec", "mean_peak_height",
    "net_slope_per_sec", "zero_cross_per_sec",
    # Spectral content (normalized — duration-invariant)
    "spectral_centroid", "spectral_bandwidth", "spectral_entropy",
    "fft_top1_rel", "fft_top2_rel", "fft_top3_rel",
    # Autocorrelation & stationarity proxies
    "autocorr_lag1", "autocorr_lag5",
    # Optional context (NOT duration — but sample count as a quality indicator)
    "n_samples",
]

def _safe(x, default=0.0):
    return default if (x is None or not np.isfinite(x)) else float(x)

def _spectral_entropy(psd):
    psd = psd[psd > 0]
    if psd.size == 0:
        return 0.0
    p = psd / psd.sum()
    return float(-(p * np.log(p)).sum())

def extract_features(values: np.ndarray, dur_sec: float) -> np.ndarray:
    """Return duration-invariant feature vector."""
    v = values[~np.isnan(values)].astype(float)
    n = len(v)
    if n < 5 or dur_sec <= 0:
        return np.full(len(FEATURE_NAMES), np.nan)

    # --- Basic stats ---
    mean_  = np.mean(v)
    std_   = np.std(v)
    median_= np.median(v)
    q25, q75 = np.percentile(v, [25, 75])
    iqr_   = q75 - q25
    sk_    = _safe(stats.skew(v))
    ku_    = _safe(stats.kurtosis(v))
    min_   = np.min(v)
    max_   = np.max(v)
    ptp_   = max_ - min_
    p05_   = np.percentile(v, 5)
    p95_   = np.percentile(v, 95)

    # --- Rate-normalized dynamics (per second → duration-invariant) ---
    energy_per_sec        = np.trapz(v * v) / dur_sec
    abs_mean_diff_per_sec = np.mean(np.abs(np.diff(v))) * (n / dur_sec) if n > 1 else 0.0

    peaks, props = signal.find_peaks(v)
    n_peaks_per_sec  = len(peaks) / dur_sec
    mean_peak_height = float(np.mean(v[peaks])) if len(peaks) > 0 else 0.0

    net_slope_per_sec = (v[-1] - v[0]) / dur_sec

    v_centered = v - mean_
    zero_cross = np.sum(np.diff(np.signbit(v_centered)))
    zero_cross_per_sec = zero_cross / dur_sec

    # --- Spectral features (normalized by signal length → shape of spectrum) ---
    v_detrend = v - mean_
    fft_mag = np.abs(rfft(v_detrend))
    if fft_mag.sum() > 0:
        freqs = np.arange(len(fft_mag))
        p = fft_mag / (fft_mag.sum() + 1e-12)
        spectral_centroid  = float(np.sum(freqs * p))
        spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * p)))
        spectral_entropy_  = _spectral_entropy(fft_mag ** 2)
        # Top-3 relative magnitudes (ignore DC, which we removed anyway)
        top3 = np.sort(fft_mag)[-3:][::-1] / (fft_mag.sum() + 1e-12)
        fft_top1_rel = float(top3[0]) if len(top3) > 0 else 0.0
        fft_top2_rel = float(top3[1]) if len(top3) > 1 else 0.0
        fft_top3_rel = float(top3[2]) if len(top3) > 2 else 0.0
    else:
        spectral_centroid = spectral_bandwidth = spectral_entropy_ = 0.0
        fft_top1_rel = fft_top2_rel = fft_top3_rel = 0.0

    # --- Autocorrelation (lag in samples, duration-invariant because normalized) ---
    def _autocorr(x, lag):
        if len(x) <= lag or np.std(x) == 0:
            return 0.0
        x0, x1 = x[:-lag], x[lag:]
        c = np.corrcoef(x0, x1)[0, 1]
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
# ============================================
# 5. Build feature matrix for a SINGLE tag from Spark (streamed)
# ============================================
from pyspark.sql import types as T
import pandas as pd

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

df_features = (
    df_ts
    .repartition("tagname", "lift_id")
    .groupBy("tagname", "lift_id")
    .applyInPandas(feature_udf, schema=feature_schema)
    .cache()
)

_ = df_features.count()

# COMMAND ----------


# ============================================
# 6. Isolation Forest per tag (tag-streamed)
# ============================================
import gc

per_tag_results = []
per_tag_summary = []
per_tag_models  = {}

for tag in sorted(available_tags):
    try:
        pdf = (
            df_features
            .filter(F.col("tagname") == tag)
            .toPandas()
        )

        if pdf.empty:
            continue

        lift_ids = pdf["lift_id"].to_numpy()
        X = pdf[FEATURE_NAMES].to_numpy(dtype=float)
    except Exception as e:
        print(f"[ERR]  {tag}: {e}")
        continue

    if len(lift_ids) < MIN_LIFTS_PER_TAG:
        print(f"[SKIP] {tag}: only {len(lift_ids)} lifts")
        continue

    valid_cols = ~np.all(np.isnan(X), axis=0)
    X_valid = X[:, valid_cols]
    used_features = [f for f, k in zip(FEATURE_NAMES, valid_cols) if k]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("iforest", IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=ISOF_CONTAMINATION,
            random_state=SEED,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_valid)

    X_imp = pipe.named_steps["imputer"].transform(X_valid)
    X_sc  = pipe.named_steps["scaler"].transform(X_imp)
    raw_scores = pipe.named_steps["iforest"].score_samples(X_sc)
    anomaly_score = -raw_scores
    is_anomaly = (pipe.named_steps["iforest"].predict(X_sc) == -1).astype(int)

    per_tag_results.append(pd.DataFrame({
        "lift_id": lift_ids,
        "tagname": tag,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
    }))
    per_tag_models[tag] = pipe
    per_tag_summary.append({
        "tagname": tag,
        "n_lifts": len(lift_ids),
        "n_features": len(used_features),
        "n_anomalies": int(is_anomaly.sum()),
        "anomaly_rate": float(is_anomaly.mean()),
        "score_mean": float(anomaly_score.mean()),
        "score_std":  float(anomaly_score.std()),
    })
    print(f"[OK]   {tag}: {len(lift_ids)} lifts, "
          f"{int(is_anomaly.sum())} anomalies")

    # Free this tag's memory before moving to the next
    del X, X_valid, X_imp, X_sc, lift_ids
    gc.collect()
# Save Spark feature table
df_features.write.mode("overwrite").format("delta").saveAsTable(
    "default.lift_tag_features_iforest"
)
# Combine anomaly results
results_df = pd.concat(per_tag_results, ignore_index=True)
summary_df = pd.DataFrame(per_tag_summary)

# Save as Spark/Delta tables
spark.createDataFrame(results_df).write.mode("overwrite").format("delta").saveAsTable(
    "default.lift_tag_iforest_scores"
)

spark.createDataFrame(summary_df).write.mode("overwrite").format("delta").saveAsTable(
    "default.lift_tag_iforest_summary"
)


# COMMAND ----------

import joblib
import os

MODEL_DIR = "/Volumes/craneds_dev/maxedge_lhdp/crane_files/singel_tag_anomali"

try:
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(per_tag_models, os.path.join(MODEL_DIR, "per_tag_iforest_models.joblib"))
    joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, "feature_names.joblib"))
    joblib.dump(tags_of_interest, os.path.join(MODEL_DIR, "tags_of_interest.joblib"))

    print(f"Models saved to: {MODEL_DIR}")

finally:
    df_ts.unpersist()

# COMMAND ----------

summary_pdf = (
    pd.DataFrame(per_tag_summary)
    .sort_values("n_anomalies", ascending=False)
    .reset_index(drop=True)
)

display(summary_pdf)

# COMMAND ----------

worst_tags = (
    pd.DataFrame(per_tag_summary)
    .sort_values("anomaly_rate", ascending=False)
    .reset_index(drop=True)
)

display(worst_tags.head(20))

# COMMAND ----------

all_results = pd.concat(per_tag_results, ignore_index=True)

# Join with df_liftlog to get movementtype for each lift_id
liftlog_cols = ["lift_id", "movementtype"]

liftlog_df = df_liftlog.select(liftlog_cols)
liftlog_df = liftlog_df.withColumn("lift_id", liftlog_df["lift_id"].cast("string"))

worst_lift_tags = (
    all_results
    .merge(liftlog_df.toPandas(), on="lift_id", how="left")
    .sort_values("anomaly_score", ascending=False)
    .reset_index(drop=True)
)

# Display only lifts that are anomalies, showing only lift_id and grouping by lift_id
display(
    worst

# COMMAND ----------

display(
    worst_lift_tags
    .merge(summary_pdf[["tagname", "n_lifts", "anomaly_rate"]], on="tagname")
)