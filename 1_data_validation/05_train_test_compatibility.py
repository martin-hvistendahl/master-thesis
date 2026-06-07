# Databricks notebook source
# Databricks / PySpark script: compare training vs overload/testing data

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import NumericType, StringType, BooleanType, TimestampType, DateType

# -----------------------------
# 1. Load data
# -----------------------------
df_raw_test = spark.table("default.df_raw_liftid_overload")
df_liftlog_test = spark.table("default.df_liftlog_overload")

df_raw_train = spark.table("default.df_all_raw_liftid")
df_liftlog_train = spark.table("default.all_liftlog")

# -----------------------------
# 2. Helpers
# -----------------------------
def numeric_cols(df):
    return [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

def categorical_cols(df):
    return [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, (StringType, BooleanType))
    ]

def timestamp_cols(df):
    return [
        f.name for f in df.schema.fields
        if isinstance(f.dataType, (TimestampType, DateType))
    ]

def basic_profile(df, dataset_name):
    return spark.createDataFrame([{
        "dataset": dataset_name,
        "rows": df.count(),
        "columns": len(df.columns),
        "distinct_lifts": df.select("lift_id").distinct().count() if "lift_id" in df.columns else None
    }])

def missing_profile(df, dataset_name):
    total = df.count()
    rows = []
    for c in df.columns:
        rows.append((
            dataset_name,
            c,
            total,
            df.filter(F.col(c).isNull()).count(),
        ))
    return spark.createDataFrame(
        rows,
        ["dataset", "column", "rows", "missing_count"]
    ).withColumn(
        "missing_pct",
        F.round(F.col("missing_count") / F.col("rows") * 100, 3)
    )

def categorical_distribution(df_train, df_test, colname):
    train = (
        df_train.groupBy(colname)
        .count()
        .withColumnRenamed("count", "train_count")
    )
    test = (
        df_test.groupBy(colname)
        .count()
        .withColumnRenamed("count", "test_count")
    )

    out = (
        train.join(test, colname, "full")
        .fillna(0, subset=["train_count", "test_count"])
        .withColumn("column", F.lit(colname))
    )

    train_total = df_train.count()
    test_total = df_test.count()

    return (
        out.withColumn("train_pct", F.round(F.col("train_count") / F.lit(train_total) * 100, 3))
           .withColumn("test_pct", F.round(F.col("test_count") / F.lit(test_total) * 100, 3))
           .select("column", colname, "train_count", "train_pct", "test_count", "test_pct")
    )

def make_numeric_bucket_expr(colname, quantiles):
    q = sorted(list(set([float(x) for x in quantiles if x is not None])))

    if len(q) < 2:
        return F.lit("single_value_or_no_bucket")

    expr = F.when(F.col(colname).isNull(), F.lit("missing"))

    for i in range(len(q) - 1):
        lower = q[i]
        upper = q[i + 1]

        label = f"[{lower:.3f}, {upper:.3f})"

        if i == len(q) - 2:
            expr = expr.when(
                (F.col(colname) >= lower) & (F.col(colname) <= upper),
                F.lit(f"[{lower:.3f}, {upper:.3f}]")
            )
        else:
            expr = expr.when(
                (F.col(colname) >= lower) & (F.col(colname) < upper),
                F.lit(label)
            )

    expr = expr.otherwise(F.lit("outside_training_range"))
    return expr

def numeric_bucket_comparison(df_train, df_test, colname, rel_error=0.01):
    # Buckets are based on training quantiles, then applied to both train and test
    quantiles = df_train.approxQuantile(
        colname,
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        rel_error
    )

    train_b = (
        df_train
        .withColumn("bucket", make_numeric_bucket_expr(colname, quantiles))
        .groupBy("bucket")
        .count()
        .withColumnRenamed("count", "train_count")
    )

    test_b = (
        df_test
        .withColumn("bucket", make_numeric_bucket_expr(colname, quantiles))
        .groupBy("bucket")
        .count()
        .withColumnRenamed("count", "test_count")
    )

    train_total = df_train.count()
    test_total = df_test.count()

    out = (
        train_b.join(test_b, "bucket", "full")
        .fillna(0, subset=["train_count", "test_count"])
        .withColumn("column", F.lit(colname))
        .withColumn("train_pct", F.col("train_count") / F.lit(train_total))
        .withColumn("test_pct", F.col("test_count") / F.lit(test_total))
        .withColumn(
            "abs_pct_diff",
            F.abs(F.col("test_pct") - F.col("train_pct"))
        )
        .withColumn(
            "psi_component",
            F.when(
                (F.col("train_pct") > 0) & (F.col("test_pct") > 0),
                (F.col("test_pct") - F.col("train_pct")) * F.log(F.col("test_pct") / F.col("train_pct"))
            ).otherwise(F.lit(0.0))
        )
        .select(
            "column",
            "bucket",
            "train_count",
            F.round(F.col("train_pct") * 100, 3).alias("train_pct"),
            "test_count",
            F.round(F.col("test_pct") * 100, 3).alias("test_pct"),
            F.round(F.col("abs_pct_diff") * 100, 3).alias("abs_pct_diff"),
            "psi_component"
        )
    )

    return out

# -----------------------------
# 3. Basic source comparison
# -----------------------------
summary = (
    basic_profile(df_raw_train, "raw_training")
    .unionByName(basic_profile(df_raw_test, "raw_overload_testing"))
    .unionByName(basic_profile(df_liftlog_train, "liftlog_training"))
    .unionByName(basic_profile(df_liftlog_test, "liftlog_overload_testing"))
)

display(summary)

# -----------------------------
# 4. Lift overlap / separation
# -----------------------------
if "lift_id" in df_liftlog_train.columns and "lift_id" in df_liftlog_test.columns:
    train_lifts = df_liftlog_train.select("lift_id").distinct()
    test_lifts = df_liftlog_test.select("lift_id").distinct()

    overlap_lifts = train_lifts.join(test_lifts, "lift_id", "inner").count()
    train_lift_count = train_lifts.count()
    test_lift_count = test_lifts.count()

    lift_overlap_summary = spark.createDataFrame([{
        "train_lifts": train_lift_count,
        "test_lifts": test_lift_count,
        "overlap_lifts": overlap_lifts,
        "test_overlap_pct": round(overlap_lifts / test_lift_count * 100, 3) if test_lift_count else None,
        "conclusion": "OK: no lift leakage" if overlap_lifts == 0 else "WARNING: train/test lift overlap exists"
    }])

    display(lift_overlap_summary)

# -----------------------------
# 5. Missing value comparison
# -----------------------------
missing_liftlog = (
    missing_profile(df_liftlog_train, "liftlog_training")
    .unionByName(missing_profile(df_liftlog_test, "liftlog_overload_testing"))
)

display(missing_liftlog.orderBy("column", "dataset"))

# -----------------------------
# 6. Raw data sensor/tag coverage
# -----------------------------
# Assumes raw data has something like sensor_tag and value.
# Rename here if your column names differ.
sensor_col_candidates = ["sensor_tag", "tag", "variable", "signal"]
value_col_candidates = ["value", "sensor_value"]

sensor_col = next((c for c in sensor_col_candidates if c in df_raw_train.columns), None)
value_col = next((c for c in value_col_candidates if c in df_raw_train.columns), None)

if sensor_col:
    raw_sensor_train = (
        df_raw_train.groupBy(sensor_col)
        .agg(
            F.count("*").alias("train_rows"),
            F.countDistinct("lift_id").alias("train_lifts") if "lift_id" in df_raw_train.columns else F.lit(None).alias("train_lifts")
        )
    )

    raw_sensor_test = (
        df_raw_test.groupBy(sensor_col)
        .agg(
            F.count("*").alias("test_rows"),
            F.countDistinct("lift_id").alias("test_lifts") if "lift_id" in df_raw_test.columns else F.lit(None).alias("test_lifts")
        )
    )

    raw_sensor_coverage = (
        raw_sensor_train.join(raw_sensor_test, sensor_col, "full")
        .fillna(0, subset=["train_rows", "test_rows", "train_lifts", "test_lifts"])
        .withColumn("in_training", F.col("train_rows") > 0)
        .withColumn("in_testing", F.col("test_rows") > 0)
        .withColumn(
            "coverage_status",
            F.when((F.col("in_training")) & (F.col("in_testing")), "present_in_both")
             .when((F.col("in_training")) & (~F.col("in_testing")), "missing_in_test")
             .when((~F.col("in_training")) & (F.col("in_testing")), "new_in_test")
             .otherwise("unknown")
        )
    )

    display(raw_sensor_coverage.orderBy("coverage_status", sensor_col))

# -----------------------------
# 7. LiftLog categorical bucket/count comparison
# -----------------------------
important_categorical = [
    c for c in ["movement_type", "crane_configuration"]
    if c in df_liftlog_train.columns and c in df_liftlog_test.columns
]

cat_results = []
for c in important_categorical:
    cat_results.append(categorical_distribution(df_liftlog_train, df_liftlog_test, c))

if cat_results:
    categorical_comparison = cat_results[0]
    for r in cat_results[1:]:
        categorical_comparison = categorical_comparison.unionByName(r, allowMissingColumns=True)

    display(categorical_comparison)

# -----------------------------
# 8. LiftLog numeric buckets
# -----------------------------
exclude_cols = {"lift_id"}
common_numeric_liftlog = [
    c for c in numeric_cols(df_liftlog_train)
    if c in df_liftlog_test.columns and c not in exclude_cols
]

numeric_bucket_tables = []
psi_rows = []

for c in common_numeric_liftlog:
    bucketed = numeric_bucket_comparison(df_liftlog_train, df_liftlog_test, c)
    numeric_bucket_tables.append(bucketed)

    psi = bucketed.agg(F.sum("psi_component").alias("psi")).collect()[0]["psi"]
    psi_rows.append((c, float(psi) if psi is not None else None))

if numeric_bucket_tables:
    numeric_bucket_comparison_all = numeric_bucket_tables[0]
    for t in numeric_bucket_tables[1:]:
        numeric_bucket_comparison_all = numeric_bucket_comparison_all.unionByName(t)

    display(numeric_bucket_comparison_all.orderBy("column", "bucket"))

    psi_summary = spark.createDataFrame(psi_rows, ["column", "psi"])
    psi_summary = psi_summary.withColumn(
        "distribution_shift",
        F.when(F.col("psi") < 0.10, "low_shift")
         .when(F.col("psi") < 0.25, "moderate_shift")
         .otherwise("large_shift")
    )

    display(psi_summary.orderBy(F.desc("psi")))

# -----------------------------
# 9. Raw value buckets per sensor/tag
# -----------------------------
# This compares value distributions for each sensor tag.
# It can be heavy if there are many tags. Limit to common tags with enough data.
if sensor_col and value_col:
    common_tags = (
        df_raw_train.groupBy(sensor_col).count().withColumnRenamed("count", "train_count")
        .join(
            df_raw_test.groupBy(sensor_col).count().withColumnRenamed("count", "test_count"),
            sensor_col,
            "inner"
        )
        .filter((F.col("train_count") >= 1000) & (F.col("test_count") >= 100))
        .orderBy(F.desc("train_count"))
        .limit(200)
        .select(sensor_col)
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    raw_bucket_results = []
    raw_psi_rows = []

    for tag in common_tags:
        tr = df_raw_train.filter(F.col(sensor_col) == tag).select(value_col)
        te = df_raw_test.filter(F.col(sensor_col) == tag).select(value_col)

        if value_col in numeric_cols(tr):
            b = numeric_bucket_comparison(tr, te, value_col)
            b = b.withColumn(sensor_col, F.lit(tag))
            raw_bucket_results.append(b)

            psi = b.agg(F.sum("psi_component").alias("psi")).collect()[0]["psi"]
            raw_psi_rows.append((tag, float(psi) if psi is not None else None))

    if raw_bucket_results:
        raw_bucket_comparison_all = raw_bucket_results[0]
        for r in raw_bucket_results[1:]:
            raw_bucket_comparison_all = raw_bucket_comparison_all.unionByName(r, allowMissingColumns=True)

        display(raw_bucket_comparison_all.orderBy(sensor_col, "bucket"))

        raw_psi_summary = spark.createDataFrame(raw_psi_rows, [sensor_col, "psi"])
        raw_psi_summary = raw_psi_summary.withColumn(
            "distribution_shift",
            F.when(F.col("psi") < 0.10, "low_shift")
             .when(F.col("psi") < 0.25, "moderate_shift")
             .otherwise("large_shift")
        )

        display(raw_psi_summary.orderBy(F.desc("psi")))

# -----------------------------
# 10. Final conclusion table
# -----------------------------
# Rules:
# - No lift overlap is best.
# - Test should contain enough lifts.
# - Most important categorical buckets should exist in both.
# - Numeric PSI should mostly be low/moderate, unless overload is intentionally shifted.
# - Raw sensors should mostly be present in both.

conclusion_items = []

if "lift_id" in df_liftlog_train.columns and "lift_id" in df_liftlog_test.columns:
    conclusion_items.append((
        "lift_leakage",
        overlap_lifts,
        "PASS" if overlap_lifts == 0 else "FAIL",
        "Training and testing should not contain the same lift_id values."
    ))

if sensor_col:
    coverage_counts = raw_sensor_coverage.groupBy("coverage_status").count().collect()
    coverage_dict = {r["coverage_status"]: r["count"] for r in coverage_counts}

    conclusion_items.append((
        "raw_sensor_present_in_both",
        coverage_dict.get("present_in_both", 0),
        "INFO",
        "Number of raw sensor tags available in both training and overload/testing data."
    ))

    conclusion_items.append((
        "raw_sensor_missing_in_test",
        coverage_dict.get("missing_in_test", 0),
        "WARN" if coverage_dict.get("missing_in_test", 0) > 0 else "PASS",
        "Sensor tags present in training but missing in test should be checked."
    ))

if numeric_bucket_tables:
    psi_counts = psi_summary.groupBy("distribution_shift").count().collect()
    psi_dict = {r["distribution_shift"]: r["count"] for r in psi_counts}

    conclusion_items.append((
        "liftlog_low_shift_variables",
        psi_dict.get("low_shift", 0),
        "INFO",
        "Low PSI means training and testing distributions are similar."
    ))

    conclusion_items.append((
        "liftlog_large_shift_variables",
        psi_dict.get("large_shift", 0),
        "WARN" if psi_dict.get("large_shift", 0) > 0 else "PASS",
        "Large PSI means overload/testing differs strongly from training. This may be expected for overload data."
    ))

final_conclusion = spark.createDataFrame(
    conclusion_items,
    ["check", "value", "status", "interpretation"]
)

display(final_conclusion)

# COMMAND ----------

from pyspark.sql import functions as F

tag_col = "tagname" if "tagname" in df_raw_train.columns else "sensor_tag"

train_total = df_raw_train.count()
test_total = df_raw_test.count()

train_counts = (
    df_raw_train
    .groupBy(tag_col)
    .agg(F.count("*").alias("train_count"))
)

test_counts = (
    df_raw_test
    .groupBy(tag_col)
    .agg(F.count("*").alias("test_count"))
)

tag_counts_comparison = (
    train_counts.join(test_counts, tag_col, "outer")
    .fillna(0, subset=["train_count", "test_count"])
    .withColumn(
        "tag_in_both",
        F.when((F.col("train_count") > 0) & (F.col("test_count") > 0), tag_col).otherwise(F.lit(None))
    )
    .withColumn("train_pct", F.round(F.col("train_count") / F.lit(train_total) * 100, 4))
    .withColumn("test_pct", F.round(F.col("test_count") / F.lit(test_total) * 100, 4))
    .orderBy(tag_col)
    .select(
        tag_col,
        "tag_in_both",
        "train_count",
        "test_count",
        "train_pct",
        "test_pct"
    )
)

display(tag_counts_comparison)

# COMMAND ----------

from pyspark.sql import functions as F

# Change if your raw tag column has another name
tag_col = "tagname" if "tagname" in df_raw_train.columns else "sensor_tag"

train_total = df_raw_train.count()
test_total = df_raw_test.count()

tag_train = (
    df_raw_train
    .groupBy(tag_col)
    .agg(
        F.count("*").alias("train_count"),
        F.countDistinct("lift_id").alias("train_lifts")
    )
    .withColumn("train_prop", F.col("train_count") / F.lit(train_total))
)

tag_test = (
    df_raw_test
    .groupBy(tag_col)
    .agg(
        F.count("*").alias("test_count"),
        F.countDistinct("lift_id").alias("test_lifts")
    )
    .withColumn("test_prop", F.col("test_count") / F.lit(test_total))
)

tag_comparison = (
    tag_train
    .join(tag_test, tag_col, "full")
    .fillna(0, subset=["train_count", "test_count", "train_lifts", "test_lifts", "train_prop", "test_prop"])
    .withColumn("train_pct", F.round(F.col("train_prop") * 100, 4))
    .withColumn("test_pct", F.round(F.col("test_prop") * 100, 4))
    .withColumn("prop_diff_pct", F.round((F.col("test_prop") - F.col("train_prop")) * 100, 4))
    .withColumn(
        "test_train_count_ratio",
        F.when(F.col("train_count") > 0, F.col("test_count") / F.col("train_count"))
         .otherwise(None)
    )
    .withColumn(
        "coverage_status",
        F.when((F.col("train_count") > 0) & (F.col("test_count") > 0), "present_in_both")
         .when((F.col("train_count") > 0) & (F.col("test_count") == 0), "missing_in_test")
         .when((F.col("train_count") == 0) & (F.col("test_count") > 0), "new_in_test")
         .otherwise("unknown")
    )
    .withColumn(
        "tag_shift_flag",
        F.when(F.col("coverage_status") != "present_in_both", "coverage_issue")
         .when(F.abs(F.col("prop_diff_pct")) >= 1.0, "large_proportion_shift")
         .when(F.abs(F.col("prop_diff_pct")) >= 0.1, "moderate_proportion_shift")
         .otherwise("similar_proportion")
    )
    .select(
        tag_col,
        "coverage_status",
        "tag_shift_flag",
        "train_count",
        "train_pct",
        "train_lifts",
        "test_count",
        "test_pct",
        "test_lifts",
        "prop_diff_pct",
        F.round("test_train_count_ratio", 6).alias("test_train_count_ratio")
    )
)

display(tag_comparison.orderBy("coverage_status", F.desc("train_count")))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from functools import reduce

# ============================================================
# 0) SETTINGS
# ============================================================

GROUP_NAME = "always_on"     # change group by group
TIMESTAMP_COL = "timestamp_utc"

RUN_TRAINING = True
RUN_TEST = True

SAMPLE_TRAINING = True
MAX_TRAINING_LIFTS = 200

OUTPUT_PREFIX = "default.group_quality"

# ============================================================
# 1) INPUT TABLES
# ============================================================

df_raw_training = spark.table("default.df_all_raw_liftid")
df_raw_test = spark.table("default.df_raw_liftid_overload")

# ============================================================
# 2) TAG GROUPS
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

tags = taggroups[GROUP_NAME]
bucket_size = bucket_sizes[GROUP_NAME]
max_gap = fill_windows[GROUP_NAME]

tag_to_col = {tag: f"tag_{i}" for i, tag in enumerate(tags)}
col_to_tag = {v: k for k, v in tag_to_col.items()}
safe_cols = list(tag_to_col.values())

print("Running group:", GROUP_NAME)
print("Number of tags:", len(tags))
print("Bucket size seconds:", bucket_size)

# ============================================================
# 3) HELPERS
# ============================================================

def safe_name(s):
    return s.replace(".", "_").replace(" ", "_").replace("-", "_").replace("/", "_")

def maybe_sample_training(raw_df, dataset_name):
    if dataset_name == "training" and SAMPLE_TRAINING:
        sample_lifts = raw_df.select("lift_id").distinct().limit(MAX_TRAINING_LIFTS)
        return raw_df.join(sample_lifts, "lift_id", "inner")
    return raw_df

def build_group_dataset(raw_df, dataset_name):
    raw_df = maybe_sample_training(raw_df, dataset_name)

    long_df = (
        raw_df
        .filter(F.col("tagname").isin(tags))
        .withColumn("dataset", F.lit(dataset_name))
        .withColumn("tag_group", F.lit(GROUP_NAME))
        .withColumn("event_ts", F.col(TIMESTAMP_COL).cast("timestamp"))
        .withColumn("ts_unix", F.unix_timestamp("event_ts"))
        .withColumn(
            "bucket_start_unix",
            (F.floor(F.col("ts_unix") / F.lit(bucket_size)) * F.lit(bucket_size)).cast("long")
        )
        .withColumn(
            "lift_group_bucket",
            F.concat_ws("_", F.col("lift_id").cast("string"), F.col("bucket_start_unix").cast("string"))
        )
    )

    pivoted = (
        long_df
        .groupBy("dataset", "tag_group", "lift_group_bucket", "bucket_start_unix")
        .pivot("tagname", tags)
        .agg(F.mean("value"))
    )

    for tag, safe in tag_to_col.items():
        if tag in pivoted.columns:
            pivoted = pivoted.withColumnRenamed(tag, safe)
        else:
            pivoted = pivoted.withColumn(safe, F.lit(None).cast("double"))

    return pivoted.select(
        "dataset",
        "tag_group",
        "lift_group_bucket",
        "bucket_start_unix",
        *safe_cols
    )

# ============================================================
# 4) SPARK-ONLY INTERPOLATION
# ============================================================

def interpolate_one_col(df, col_name, original_tag):
    w_prev = (
        Window
        .partitionBy("dataset", "tag_group", "lift_id")
        .orderBy("bucket_start_unix")
        .rowsBetween(Window.unboundedPreceding, 0)
    )

    w_next = (
        Window
        .partitionBy("dataset", "tag_group", "lift_id")
        .orderBy("bucket_start_unix")
        .rowsBetween(0, Window.unboundedFollowing)
    )

    prev_val = F.last(F.when(F.col(col_name).isNotNull(), F.col(col_name)), ignorenulls=True).over(w_prev)
    prev_ts = F.last(F.when(F.col(col_name).isNotNull(), F.col("bucket_start_unix")), ignorenulls=True).over(w_prev)

    next_val = F.first(F.when(F.col(col_name).isNotNull(), F.col(col_name)), ignorenulls=True).over(w_next)
    next_ts = F.first(F.when(F.col(col_name).isNotNull(), F.col("bucket_start_unix")), ignorenulls=True).over(w_next)

    gap_buckets = ((next_ts - prev_ts) / F.lit(bucket_size) - F.lit(1))

    linear_value = (
        prev_val +
        ((next_val - prev_val) *
         ((F.col("bucket_start_unix") - prev_ts) / (next_ts - prev_ts)))
    )

    zero_fill = (
        (original_tag in zero_fill_columns)
        and True
    )

    if zero_fill:
        filled_expr = (
            F.when(F.col(col_name).isNotNull(), F.col(col_name))
             .when(
                 (prev_val == 0) &
                 (next_val == 0) &
                 prev_ts.isNotNull() &
                 next_ts.isNotNull() &
                 (gap_buckets <= max_gap),
                 F.lit(0.0)
             )
             .when(
                 prev_val.isNotNull() &
                 next_val.isNotNull() &
                 (gap_buckets <= max_gap),
                 linear_value
             )
             .otherwise(F.col(col_name))
        )
    else:
        filled_expr = (
            F.when(F.col(col_name).isNotNull(), F.col(col_name))
             .when(
                 prev_val.isNotNull() &
                 next_val.isNotNull() &
                 (gap_buckets <= max_gap),
                 linear_value
             )
             .otherwise(F.col(col_name))
        )

    return df.withColumn(col_name, filled_expr.cast("double"))

def interpolate_group_spark(df):
    df = df.withColumn(
        "lift_id",
        F.regexp_extract(F.col("lift_group_bucket"), r"^([^_]+)_", 1)
    )

    for original_tag, safe_col in tag_to_col.items():
        df = interpolate_one_col(df, safe_col, original_tag)

    return df.drop("lift_id")

# ============================================================
# 5) STATS
# ============================================================

def tag_stats(df, dataset_name, stage_name):
    total_rows = df.count()
    rows = []

    for original_tag, safe_col in tag_to_col.items():
        r = df.agg(
            F.count(F.col(safe_col)).alias("non_null"),
            F.count(F.when(F.col(safe_col).isNull(), 1)).alias("nulls"),
            F.min(F.col(safe_col)).alias("min"),
            F.max(F.col(safe_col)).alias("max"),
            F.mean(F.col(safe_col)).alias("mean"),
            F.stddev(F.col(safe_col)).alias("stddev"),
        ).collect()[0]

        rows.append((
            dataset_name,
            GROUP_NAME,
            stage_name,
            original_tag,
            total_rows,
            int(r["non_null"]),
            int(r["nulls"]),
            round(100 * r["nulls"] / total_rows, 3) if total_rows else None,
            float(r["min"]) if r["min"] is not None else None,
            float(r["max"]) if r["max"] is not None else None,
            float(r["mean"]) if r["mean"] is not None else None,
            float(r["stddev"]) if r["stddev"] is not None else None,
        ))

    return spark.createDataFrame(
        rows,
        [
            "dataset",
            "tag_group",
            "stage",
            "tagname",
            "rows",
            "non_null_count",
            "null_count",
            "null_pct",
            "min",
            "max",
            "mean",
            "stddev",
        ]
    )

def lift_quality(df, dataset_name, stage_name):
    present_expr = reduce(
        lambda a, b: a + b,
        [F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in safe_cols]
    )

    missing_expr = reduce(
        lambda a, b: a + b,
        [F.when(F.col(c).isNull(), 1).otherwise(0) for c in safe_cols]
    )

    return (
        df
        .withColumn("lift_id", F.regexp_extract(F.col("lift_group_bucket"), r"^([^_]+)_", 1))
        .withColumn("present_values", present_expr)
        .withColumn("missing_values", missing_expr)
        .groupBy("dataset", "tag_group", "lift_id")
        .agg(
            F.count("*").alias("n_buckets"),
            F.min("bucket_start_unix").alias("first_bucket"),
            F.max("bucket_start_unix").alias("last_bucket"),
            F.sum("present_values").alias("observed_values"),
            F.sum("missing_values").alias("missing_values"),
        )
        .withColumn("stage", F.lit(stage_name))
        .withColumn("n_tags", F.lit(len(tags)))
        .withColumn("expected_values", F.col("n_buckets") * F.col("n_tags"))
        .withColumn("duration_min", F.round((F.col("last_bucket") - F.col("first_bucket")) / 60, 2))
        .withColumn("coverage_pct", F.round(F.col("observed_values") / F.col("expected_values") * 100, 2))
        .withColumn("missing_pct", F.round(F.col("missing_values") / F.col("expected_values") * 100, 2))
        .withColumn(
            "lift_quality_flag",
            F.when(F.col("coverage_pct") >= 95, "excellent")
             .when(F.col("coverage_pct") >= 80, "good")
             .when(F.col("coverage_pct") >= 60, "usable_with_caution")
             .otherwise("poor")
        )
        .select(
            "dataset",
            "tag_group",
            "stage",
            "lift_id",
            "n_tags",
            "n_buckets",
            "duration_min",
            "expected_values",
            "observed_values",
            "missing_values",
            "coverage_pct",
            "missing_pct",
            "lift_quality_flag",
        )
    )

# ============================================================
# 6) RUN
# ============================================================

before_stats_dfs = []
after_stats_dfs = []
before_lift_quality_dfs = []
after_lift_quality_dfs = []

if RUN_TRAINING:
    print("Building training group:", GROUP_NAME)
    train_pivot = build_group_dataset(df_raw_training, "training").cache()
    print("Training rows:", train_pivot.count())

    before_stats_dfs.append(tag_stats(train_pivot, "training", "before_interpolation"))
    before_lift_quality_dfs.append(lift_quality(train_pivot, "training", "before_interpolation"))

    print("Interpolating training group:", GROUP_NAME)
    train_filled = interpolate_group_spark(train_pivot).cache()
    print("Training filled rows:", train_filled.count())

    after_stats_dfs.append(tag_stats(train_filled, "training", "after_interpolation"))
    after_lift_quality_dfs.append(lift_quality(train_filled, "training", "after_interpolation"))

if RUN_TEST:
    print("Building test_overload group:", GROUP_NAME)
    test_pivot = build_group_dataset(df_raw_test, "test_overload").cache()
    print("Test rows:", test_pivot.count())

    before_stats_dfs.append(tag_stats(test_pivot, "test_overload", "before_interpolation"))
    before_lift_quality_dfs.append(lift_quality(test_pivot, "test_overload", "before_interpolation"))

    print("Interpolating test_overload group:", GROUP_NAME)
    test_filled = interpolate_group_spark(test_pivot).cache()
    print("Test filled rows:", test_filled.count())

    after_stats_dfs.append(tag_stats(test_filled, "test_overload", "after_interpolation"))
    after_lift_quality_dfs.append(lift_quality(test_filled, "test_overload", "after_interpolation"))

# ============================================================
# 7) COMBINE
# ============================================================

before_stats = reduce(lambda a, b: a.unionByName(b), before_stats_dfs)
after_stats = reduce(lambda a, b: a.unionByName(b), after_stats_dfs)
tag_stats_all = before_stats.unionByName(after_stats)

before_lift_quality = reduce(lambda a, b: a.unionByName(b), before_lift_quality_dfs)
after_lift_quality = reduce(lambda a, b: a.unionByName(b), after_lift_quality_dfs)
lift_quality_all = before_lift_quality.unionByName(after_lift_quality)

tag_before_after = (
    before_stats.alias("b")
    .join(
        after_stats.alias("a"),
        on=["dataset", "tag_group", "tagname"],
        how="inner"
    )
    .select(
        "dataset",
        "tag_group",
        "tagname",
        F.col("b.rows").alias("rows_before"),
        F.col("a.rows").alias("rows_after"),
        F.col("b.null_pct").alias("null_pct_before"),
        F.col("a.null_pct").alias("null_pct_after"),
        F.round(F.col("b.null_pct") - F.col("a.null_pct"), 3).alias("null_pct_reduction"),
        F.col("b.mean").alias("mean_before"),
        F.col("a.mean").alias("mean_after"),
        F.col("b.stddev").alias("stddev_before"),
        F.col("a.stddev").alias("stddev_after"),
    )
)

lift_before_after = (
    before_lift_quality.alias("b")
    .join(
        after_lift_quality.alias("a"),
        on=["dataset", "tag_group", "lift_id"],
        how="inner"
    )
    .select(
        "dataset",
        "tag_group",
        "lift_id",
        F.col("b.n_buckets").alias("n_buckets_before"),
        F.col("a.n_buckets").alias("n_buckets_after"),
        F.col("b.duration_min").alias("duration_min"),
        F.col("b.coverage_pct").alias("coverage_pct_before"),
        F.col("a.coverage_pct").alias("coverage_pct_after"),
        F.round(F.col("a.coverage_pct") - F.col("b.coverage_pct"), 3).alias("coverage_pct_gain"),
        F.col("b.missing_pct").alias("missing_pct_before"),
        F.col("a.missing_pct").alias("missing_pct_after"),
        F.round(F.col("b.missing_pct") - F.col("a.missing_pct"), 3).alias("missing_pct_reduction"),
        F.col("a.lift_quality_flag").alias("lift_quality_after"),
    )
)

training_vs_test_after = (
    after_stats.alias("tr")
    .filter(F.col("dataset") == "training")
    .join(
        after_stats.alias("te").filter(F.col("dataset") == "test_overload"),
        on=["tag_group", "tagname"],
        how="outer"
    )
    .select(
        "tag_group",
        "tagname",
        F.col("tr.null_pct").alias("training_null_pct_after"),
        F.col("te.null_pct").alias("test_null_pct_after"),
        F.round(F.col("te.null_pct") - F.col("tr.null_pct"), 3).alias("test_minus_training_null_pct"),
        F.col("tr.mean").alias("training_mean_after"),
        F.col("te.mean").alias("test_mean_after"),
        F.col("tr.stddev").alias("training_stddev_after"),
        F.col("te.stddev").alias("test_stddev_after"),
    )
)

# ============================================================
# 8) DISPLAY
# ============================================================

display(tag_stats_all.orderBy("dataset", "stage", F.desc("null_pct")))
display(tag_before_after.orderBy("dataset", F.desc("null_pct_before")))
display(lift_quality_all.orderBy("dataset", "stage", "coverage_pct"))
display(lift_before_after.orderBy("dataset", "coverage_pct_after"))
display(training_vs_test_after.orderBy(F.desc(F.abs(F.col("test_minus_training_null_pct")))))

# ============================================================
# 9) SAVE TABLES
# ============================================================

group_safe = safe_name(GROUP_NAME)

tag_stats_table = f"{OUTPUT_PREFIX}_{group_safe}_tag_stats"
tag_before_after_table = f"{OUTPUT_PREFIX}_{group_safe}_tag_before_after"
lift_quality_table = f"{OUTPUT_PREFIX}_{group_safe}_lift_quality"
lift_before_after_table = f"{OUTPUT_PREFIX}_{group_safe}_lift_before_after"
training_vs_test_table = f"{OUTPUT_PREFIX}_{group_safe}_training_vs_test_after"

tag_stats_all.write.mode("overwrite").saveAsTable(tag_stats_table)
tag_before_after.write.mode("overwrite").saveAsTable(tag_before_after_table)
lift_quality_all.write.mode("overwrite").saveAsTable(lift_quality_table)
lift_before_after.write.mode("overwrite").saveAsTable(lift_before_after_table)
training_vs_test_after.write.mode("overwrite").saveAsTable(training_vs_test_table)

print("Saved tables:")
print(tag_stats_table)
print(tag_before_after_table)
print(lift_quality_table)
print(lift_before_after_table)
print(training_vs_test_table)

# COMMAND ----------

# ============================================================
# RUN ALL GROUPS SAFELY, ONE BY ONE
# ============================================================

groups_to_run = [
    "always_on",
    "boom_active",
    "hoist_active",
    "hyd_common",
    "slew_active",
    "slew_power",
    "thermal_drive",
    "thermal_motor",
    "thermal_resistor",
]

RUN_TRAINING = True
RUN_TEST = True

SAMPLE_TRAINING = True
MAX_TRAINING_LIFTS = 200

OUTPUT_PREFIX = "default.group_quality"

all_tag_stats_tables = []
all_tag_before_after_tables = []
all_lift_quality_tables = []
all_lift_before_after_tables = []
all_training_vs_test_tables = []

for GROUP_NAME in groups_to_run:

    print("\n" + "=" * 80)
    print(f"RUNNING GROUP: {GROUP_NAME}")
    print("=" * 80)

    tags = taggroups[GROUP_NAME]
    bucket_size = bucket_sizes[GROUP_NAME]
    max_gap = fill_windows[GROUP_NAME]

    tag_to_col = {tag: f"tag_{i}" for i, tag in enumerate(tags)}
    col_to_tag = {v: k for k, v in tag_to_col.items()}
    safe_cols = list(tag_to_col.values())

    print("Number of tags:", len(tags))
    print("Bucket size seconds:", bucket_size)
    print("Max interpolation gap:", max_gap)

    before_stats_dfs = []
    after_stats_dfs = []
    before_lift_quality_dfs = []
    after_lift_quality_dfs = []

    if RUN_TRAINING:
        print(f"\nBuilding training group: {GROUP_NAME}")
        train_pivot = build_group_dataset(df_raw_training, "training").cache()
        print("Training rows:", train_pivot.count())

        before_stats_dfs.append(
            tag_stats(train_pivot, "training", "before_interpolation")
        )

        before_lift_quality_dfs.append(
            lift_quality(train_pivot, "training", "before_interpolation")
        )

        print(f"Interpolating training group: {GROUP_NAME}")
        train_filled = interpolate_group_spark(train_pivot).cache()
        print("Training filled rows:", train_filled.count())

        after_stats_dfs.append(
            tag_stats(train_filled, "training", "after_interpolation")
        )

        after_lift_quality_dfs.append(
            lift_quality(train_filled, "training", "after_interpolation")
        )

        train_pivot.unpersist()

    if RUN_TEST:
        print(f"\nBuilding test_overload group: {GROUP_NAME}")
        test_pivot = build_group_dataset(df_raw_test, "test_overload").cache()
        print("Test rows:", test_pivot.count())

        before_stats_dfs.append(
            tag_stats(test_pivot, "test_overload", "before_interpolation")
        )

        before_lift_quality_dfs.append(
            lift_quality(test_pivot, "test_overload", "before_interpolation")
        )

        print(f"Interpolating test_overload group: {GROUP_NAME}")
        test_filled = interpolate_group_spark(test_pivot).cache()
        print("Test filled rows:", test_filled.count())

        after_stats_dfs.append(
            tag_stats(test_filled, "test_overload", "after_interpolation")
        )

        after_lift_quality_dfs.append(
            lift_quality(test_filled, "test_overload", "after_interpolation")
        )

        test_pivot.unpersist()

    before_stats = reduce(lambda a, b: a.unionByName(b), before_stats_dfs)
    after_stats = reduce(lambda a, b: a.unionByName(b), after_stats_dfs)

    tag_stats_all = before_stats.unionByName(after_stats)

    before_lift_quality = reduce(lambda a, b: a.unionByName(b), before_lift_quality_dfs)
    after_lift_quality = reduce(lambda a, b: a.unionByName(b), after_lift_quality_dfs)

    lift_quality_all = before_lift_quality.unionByName(after_lift_quality)

    tag_before_after = (
        before_stats.alias("b")
        .join(
            after_stats.alias("a"),
            on=["dataset", "tag_group", "tagname"],
            how="inner",
        )
        .select(
            "dataset",
            "tag_group",
            "tagname",
            F.col("b.rows").alias("rows_before"),
            F.col("a.rows").alias("rows_after"),
            F.col("b.null_pct").alias("null_pct_before"),
            F.col("a.null_pct").alias("null_pct_after"),
            F.round(F.col("b.null_pct") - F.col("a.null_pct"), 3).alias("null_pct_reduction"),
            F.col("b.mean").alias("mean_before"),
            F.col("a.mean").alias("mean_after"),
            F.col("b.stddev").alias("stddev_before"),
            F.col("a.stddev").alias("stddev_after"),
        )
    )

    lift_before_after = (
        before_lift_quality.alias("b")
        .join(
            after_lift_quality.alias("a"),
            on=["dataset", "tag_group", "lift_id"],
            how="inner",
        )
        .select(
            "dataset",
            "tag_group",
            "lift_id",
            F.col("b.n_buckets").alias("n_buckets_before"),
            F.col("a.n_buckets").alias("n_buckets_after"),
            F.col("b.duration_min").alias("duration_min"),
            F.col("b.coverage_pct").alias("coverage_pct_before"),
            F.col("a.coverage_pct").alias("coverage_pct_after"),
            F.round(F.col("a.coverage_pct") - F.col("b.coverage_pct"), 3).alias("coverage_pct_gain"),
            F.col("b.missing_pct").alias("missing_pct_before"),
            F.col("a.missing_pct").alias("missing_pct_after"),
            F.round(F.col("b.missing_pct") - F.col("a.missing_pct"), 3).alias("missing_pct_reduction"),
            F.col("a.lift_quality_flag").alias("lift_quality_after"),
        )
    )

    training_vs_test_after = (
        after_stats.alias("tr")
        .filter(F.col("dataset") == "training")
        .join(
            after_stats.alias("te").filter(F.col("dataset") == "test_overload"),
            on=["tag_group", "tagname"],
            how="outer",
        )
        .select(
            "tag_group",
            "tagname",
            F.col("tr.null_pct").alias("training_null_pct_after"),
            F.col("te.null_pct").alias("test_null_pct_after"),
            F.round(
                F.col("te.null_pct") - F.col("tr.null_pct"),
                3
            ).alias("test_minus_training_null_pct"),
            F.col("tr.mean").alias("training_mean_after"),
            F.col("te.mean").alias("test_mean_after"),
            F.col("tr.stddev").alias("training_stddev_after"),
            F.col("te.stddev").alias("test_stddev_after"),
        )
    )

    group_safe = safe_name(GROUP_NAME)

    tag_stats_table = f"{OUTPUT_PREFIX}_{group_safe}_tag_stats"
    tag_before_after_table = f"{OUTPUT_PREFIX}_{group_safe}_tag_before_after"
    lift_quality_table = f"{OUTPUT_PREFIX}_{group_safe}_lift_quality"
    lift_before_after_table = f"{OUTPUT_PREFIX}_{group_safe}_lift_before_after"
    training_vs_test_table = f"{OUTPUT_PREFIX}_{group_safe}_training_vs_test_after"

    tag_stats_all.write.mode("overwrite").saveAsTable(tag_stats_table)
    tag_before_after.write.mode("overwrite").saveAsTable(tag_before_after_table)
    lift_quality_all.write.mode("overwrite").saveAsTable(lift_quality_table)
    lift_before_after.write.mode("overwrite").saveAsTable(lift_before_after_table)
    training_vs_test_after.write.mode("overwrite").saveAsTable(training_vs_test_table)

    all_tag_stats_tables.append(tag_stats_table)
    all_tag_before_after_tables.append(tag_before_after_table)
    all_lift_quality_tables.append(lift_quality_table)
    all_lift_before_after_tables.append(lift_before_after_table)
    all_training_vs_test_tables.append(training_vs_test_table)

    print("\nSaved:")
    print(tag_stats_table)
    print(tag_before_after_table)
    print(lift_quality_table)
    print(lift_before_after_table)
    print(training_vs_test_table)

    if RUN_TRAINING:
        train_filled.unpersist()

    if RUN_TEST:
        test_filled.unpersist()

print("\n" + "=" * 80)
print("ALL GROUPS FINISHED")
print("=" * 80)

print("Tag stats tables:")
for t in all_tag_stats_tables:
    print(t)

print("Training vs test tables:")
for t in all_training_vs_test_tables:
    print(t)

# COMMAND ----------

from functools import reduce
from pyspark.sql import functions as F

# ============================================================
# COMBINE ALL GROUP TABLES
# ============================================================

all_tag_stats_tables = [
    "default.group_quality_always_on_tag_stats",
    "default.group_quality_boom_active_tag_stats",
    "default.group_quality_hoist_active_tag_stats",
    "default.group_quality_hyd_common_tag_stats",
    "default.group_quality_slew_active_tag_stats",
    "default.group_quality_slew_power_tag_stats",
    "default.group_quality_thermal_drive_tag_stats",
    "default.group_quality_thermal_motor_tag_stats",
    "default.group_quality_thermal_resistor_tag_stats",
]

all_tag_before_after_tables = [
    "default.group_quality_always_on_tag_before_after",
    "default.group_quality_boom_active_tag_before_after",
    "default.group_quality_hoist_active_tag_before_after",
    "default.group_quality_hyd_common_tag_before_after",
    "default.group_quality_slew_active_tag_before_after",
    "default.group_quality_slew_power_tag_before_after",
    "default.group_quality_thermal_drive_tag_before_after",
    "default.group_quality_thermal_motor_tag_before_after",
    "default.group_quality_thermal_resistor_tag_before_after",
]

all_lift_quality_tables = [
    "default.group_quality_always_on_lift_quality",
    "default.group_quality_boom_active_lift_quality",
    "default.group_quality_hoist_active_lift_quality",
    "default.group_quality_hyd_common_lift_quality",
    "default.group_quality_slew_active_lift_quality",
    "default.group_quality_slew_power_lift_quality",
    "default.group_quality_thermal_drive_lift_quality",
    "default.group_quality_thermal_motor_lift_quality",
    "default.group_quality_thermal_resistor_lift_quality",
]

all_lift_before_after_tables = [
    "default.group_quality_always_on_lift_before_after",
    "default.group_quality_boom_active_lift_before_after",
    "default.group_quality_hoist_active_lift_before_after",
    "default.group_quality_hyd_common_lift_before_after",
    "default.group_quality_slew_active_lift_before_after",
    "default.group_quality_slew_power_lift_before_after",
    "default.group_quality_thermal_drive_lift_before_after",
    "default.group_quality_thermal_motor_lift_before_after",
    "default.group_quality_thermal_resistor_lift_before_after",
]

all_training_vs_test_tables = [
    "default.group_quality_always_on_training_vs_test_after",
    "default.group_quality_boom_active_training_vs_test_after",
    "default.group_quality_hoist_active_training_vs_test_after",
    "default.group_quality_hyd_common_training_vs_test_after",
    "default.group_quality_slew_active_training_vs_test_after",
    "default.group_quality_slew_power_training_vs_test_after",
    "default.group_quality_thermal_drive_training_vs_test_after",
    "default.group_quality_thermal_motor_training_vs_test_after",
    "default.group_quality_thermal_resistor_training_vs_test_after",
]

def union_tables(table_names):
    dfs = [spark.table(t) for t in table_names]
    return reduce(lambda a, b: a.unionByName(b), dfs)

all_tag_stats = union_tables(all_tag_stats_tables)

all_tag_before_after = union_tables(all_tag_before_after_tables)

all_lift_quality = union_tables(all_lift_quality_tables)

all_lift_before_after = union_tables(all_lift_before_after_tables)

all_training_vs_test_after = union_tables(all_training_vs_test_tables)

# ============================================================
# SAVE MASTER TABLES
# ============================================================

all_tag_stats.write.mode("overwrite").saveAsTable(
    "default.group_quality_ALL_tag_stats"
)

all_tag_before_after.write.mode("overwrite").saveAsTable(
    "default.group_quality_ALL_tag_before_after"
)

all_lift_quality.write.mode("overwrite").saveAsTable(
    "default.group_quality_ALL_lift_quality"
)

all_lift_before_after.write.mode("overwrite").saveAsTable(
    "default.group_quality_ALL_lift_before_after"
)

all_training_vs_test_after.write.mode("overwrite").saveAsTable(
    "default.group_quality_ALL_training_vs_test_after"
)

print("MASTER TABLES SAVED")

# COMMAND ----------

display(
    all_tag_before_after.orderBy(
        "tag_group",
        "dataset",
        F.desc("null_pct_reduction")
    )
)

display(
    all_training_vs_test_after.orderBy(
        "tag_group",
        F.desc(F.abs(F.col("test_minus_training_null_pct")))
    )
)

display(
    all_lift_before_after.orderBy(
        "tag_group",
        "dataset",
        F.desc("coverage_pct_gain")
    )
)

# COMMAND ----------

