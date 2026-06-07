# Offshore Crane Sensor Data – Anomaly Detection Pipeline

Code accompanying the master's thesis **"Structuring Offshore Crane Sensor Data for
Machine Learning-Based Anomaly Detection"** (NTNU, Department of Information Security
and Communication Technology).

This repository contains **only the scripts that produce the results reported in the
thesis**. Exploratory notebooks and abandoned approaches have been left out (see
[What was deliberately excluded](#what-was-deliberately-excluded) at the bottom).

The goal of this repo is to let the **next student pick up the work**: it documents,
for each script, what it reads, what it writes, and where it appears in the thesis.

---

## 1. Environment

All scripts are **Databricks notebooks** (each starts with `# Databricks notebook source`
and uses `spark`, `dbutils`, and `display`). They are stored here as `.py` for version
control. To run them, import the `.py` files into a Databricks workspace, or copy the
cells (`# COMMAND ----------` marks each cell boundary).

- **Cluster:** PySpark + Python 3
- **Python libraries:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `joblib`; the LSTM
  script additionally needs `tensorflow` (installed in-notebook with `%pip install tensorflow`).
- **Data:** proprietary NOV crane export (raw sensor CSVs, lift logs, alarm logs). The
  data is **not** included in this repo and is not public.

> Paths such as `/Volumes/craneds_dev/maxedge_lhdp/crane_files/...` and Spark table
> names such as `default.df_all_raw_liftid` are environment-specific. A new user must
> update these to match their own workspace.

---

## 2. How the pipeline fits together

The pipeline has three stages. Each stage **writes artifacts that the next stage reads**,
so they must be run in order the first time.

```
 RAW NOV EXPORT (zips of CSVs)
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 0 — Setup & database  (0_setup_and_database/)          │
 │   01 unzip & sort → 02 load into Spark tables                │
 │   03 lift summary (dataset scale)                            │
 │   04 tag co-activity → the 9 signal groups                   │
 │   WRITES: Spark tables  default.df_all_raw_liftid,           │
 │           default.all_liftlog, default.df_alarms, ...        │
 └─────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 1 — Data validation  (1_data_validation/)             │
 │   05 train vs. overload-test compatibility check            │
 │   USES the tables from Stage 0; decides which tags are       │
 │   safe to model (drops extreme train/test imbalance)         │
 └─────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ STAGE 2 — Models  (2_models/)                               │
 │   For each method:  train_*  →  saves model to MODEL_DIR     │
 │                     verify_* →  loads model, scores overload │
 │   Methods: IF-interpolation, IF-nullrow,                     │
 │            single-tag IF, LSTM autoencoder                   │
 └─────────────────────────────────────────────────────────────┘
        │
        ▼
   Lift-level anomaly rankings, per-group rates, driver summaries
   (these are the tables in thesis Chapter 5)
```

**Shared preprocessing.** The three "stacked-generation" model scripts
(IF-interpolation, IF-nullrow, LSTM) each begin with the *same* preprocessing block:
the 9 activity-based tag groups, the group-specific bucket sizes, pivoting to wide
format, and rule-based missing-value handling (zero-fill + distance-weighted
interpolation). This is intentional — each notebook is self-contained so it can be run
on its own. If you refactor, this block is the natural thing to pull into a shared module.

---

## 3. Run order and what each script does

### Stage 0 — `0_setup_and_database/`

| # | File | What it does | Thesis |
|---|------|--------------|--------|
| 01 | `01_unzip_and_sort_logfiles.py` | Unzips the raw NOV export and sorts the CSVs into categories (`raw`, `liftlog`, `alarms`, `eventlog`, `annotations`). | §2.7 |
| 02 | `02_load_data_into_tables.py` | Reads the categorized CSVs into Spark and writes the core tables used everywhere downstream (raw-with-lift-id, lift log, alarms). Also checks movement-type distribution. | §2.7, §4.2 |
| 03 | `03_lift_summary_and_dataset_scale.py` | Computes dataset-scale statistics (lift count, tag count, observations per lift) and the tag-frequency / subsystem breakdowns. Produces the values exported to `thesis_values/`. | §2.7, Tables 2.1–2.2, Figs 2.3–2.6 |
| 04 | `04_tag_coactivity_grouping.py` | Builds the per-bucket presence matrix, computes **Jaccard co-activity** between tags, runs **hierarchical agglomerative clustering**, and derives the **9 activity-based signal groups** + per-group bucket sizes. This is the core grouping contribution. | §3.4.3, §4.4, Tables 4.3–4.5 |

### Stage 1 — `1_data_validation/`

| # | File | What it does | Thesis |
|---|------|--------------|--------|
| 05 | `05_train_test_compatibility.py` | Compares the **training crane** against the **overload-test crane** after preprocessing: missing-value %, distributions, tag-logging imbalance. Identifies tags to exclude so models detect crane behaviour, not logging differences. | §4.6, §5.1, Tables 4.6–4.7, 5.1–5.3 |

### Stage 2 — `2_models/`

Each method has a **train** script (fits per-group specialist models + a meta-model and
saves them) and a **verify** script (loads the saved models and scores the overload-test
data → the rankings in Chapter 5).

| Folder | Train → Verify | Method | Saves to | Thesis |
|--------|----------------|--------|----------|--------|
| `isolation_forest_interpolation/` | `train_…` → `verify_…` | Isolation Forest on grouped, interpolated windows (full representation) | `anomaly_model/` | §3.6.2, §5.3 |
| `isolation_forest_nullrow/` | `train_…` → `verify_…` | Isolation Forest on complete rows only (no interpolation) | `anomaly_model_null_rows/` | §3.6.2, §5.4 |
| `single_tag_isolation_forest/` | `train_…` → `verify_…` | One Isolation Forest **per tag** (interpretable, signal-level) | `singel_tag_anomali/` | §3.6.2, §5.5 |
| `lstm_autoencoder/` | `train_…` → `verify_…` | Sequence-to-sequence LSTM autoencoder; reconstruction error = anomaly score | `anomaly_model_lstm_autoencoder/` | §2.4.2, §3.6.3, §5.6 |

**Model artifacts (per group, written by the train scripts):** `config.json`,
`scaler_<group>.joblib`, `specialist_<group>.joblib`, and `meta_model.joblib`. The
verify scripts read exactly these, so keep the file names and column order consistent
if you retrain.

**Thresholding.** Anomaly flags use a **robust IQR-based lower bound** on the score
distribution (not a fixed contamination fraction), matching thesis §3.6.5.

---

## 4. The 9 signal groups (reference)

Defined in `04_tag_coactivity_grouping.py` and reused by every model script
(thesis Table 4.3):

| Group | Bucket size ∆t | Description |
|-------|----------------|-------------|
| `always_on` | 5 s | Baseline signals logged throughout every lift |
| `hoist_active` | 5 s | Active during hoist motion |
| `boom_active` | 5 s | Active during boom motion |
| `slew_active` | 5 s | Active during slew motion |
| `hyd_common` | 10 s | Brake pressures, active across most lift phases |
| `thermal_drive` | 30 s | Drive-side depletion-layer temperatures |
| `slew_power` | 60 s | Sparse slew power / torque-limit events |
| `thermal_resistor` | 120 s | Brake resistor & coolant temperatures |
| `thermal_motor` | 300 s | Motor temperatures |

---

## 5. `thesis_values/`

Generated numbers, tables, and TikZ figure data produced by Stage 0 (mainly
`03_lift_summary_and_dataset_scale.py`) and pasted into the Overleaf thesis. Kept here
as **provenance** linking the code to the numbers in the document — not required to run
the pipeline. (`.tex` = figures/tables, `.tsv` = underlying values, `summary_values.txt`
= scalar values cited in text.)

---

## 6. Suggested next steps (from thesis §6.6 Future work)

- Connect anomaly rankings to **stronger ground truth** (maintenance/inspection records).
- Train **movement-type-specific** models and thresholds.
- Calibrate thresholds — the LSTM currently flags almost everything, so it is useful for
  *ranking* but not yet for operational decisions.
- Investigate models that handle irregular sampling natively, to reduce reliance on
  bucketing/interpolation assumptions.

---

## What was deliberately excluded

These existed in the working folder but are **not part of the final thesis results**, so
they were left out to keep the handover clean:

- **PCA on null rows** (`PCA NULL.py`, `PCA NULL Verifiction.py`) — PCA reconstruction
  variant; not reported in Chapter 5.
- **LOF on lift-log** (`LOF on liftlog.py`) — Local Outlier Factor on lift-log summary
  stats; not in the thesis.
- **`Big boot.py`** — large scratch notebook (early alarm-based supervised
  experiments); superseded by the unsupervised pipeline above.
- **`Testing out idees/`** — early experiments (bucket testing, brute-force grouping,
  median/zero fill trials, simple regression, pivot tests).
- **`Initial data testing.py`, `Testing & Experiments on Overload crane data.py`** —
  one-off data exploration.

If you need any of these for context, they remain in the original thesis archive.
