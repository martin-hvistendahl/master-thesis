# Databricks notebook source
import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from itertools import chain
from pyspark.sql.functions import when

df_raw_with_lift = spark.table("default.df_all_raw_liftid")
df_alarms_with_lift_fit = spark.table("default.df_alarms_fit")
df_liftlog = spark.table("default.all_liftlog")
df_alarms_with_lift = spark.table("default.df_alarms")


df_liftlog_5 = df_liftlog.filter(F.col("movementtype") == 5)
df_liftlog = df_liftlog.filter(F.col("movementtype") != 5)


#laster inn litt mindre. Bare fra EQ-22057-101A_2506302159_2507142159 for å gjøre ting litt raskere
# df_raw_with_lift = spark.table("default.df_clean_with_liftid")
# df_alarms_with_lift_fit = spark.table("default.df_alarms_fit")
# df_liftlog = spark.table("default.df_liftlog")
# df_alarms_with_lift = spark.table("default.df_alarms")


pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

# ============================================================
# STEP 1: Measure each tag's median update gap (seconds)
# ============================================================

df_tag_stats = (
    df_raw_with_lift
    .filter(F.col("tagname").isin(tagnames))
    .withColumn("ts_unix", F.unix_timestamp(F.col("timestamp_utc").cast("timestamp")))
)

w = Window.partitionBy("tagname").orderBy("ts_unix")

df_gaps = (
    df_tag_stats
    .withColumn("prev_ts", F.lag("ts_unix").over(w))
    .withColumn("gap_s", F.col("ts_unix") - F.col("prev_ts"))
    .filter(F.col("gap_s").isNotNull() & (F.col("gap_s") > 0))
)

tag_freq = (
    df_gaps.groupBy("tagname")
    .agg(
        F.expr("percentile_approx(gap_s, 0.5)").alias("median_gap_s"),
        F.expr("percentile_approx(gap_s, 0.9)").alias("p90_gap_s"),
        F.count("*").alias("n_samples"),
    )
    .toPandas()
)

tag_freq["tag_short"] = tag_freq["tagname"].str.replace("ImportS7Device.", "", regex=False)
tag_freq = tag_freq.sort_values("median_gap_s").reset_index(drop=True)

print("=== ALL tag update frequencies (sorted by median gap) ===")
print(tag_freq[["tag_short", "median_gap_s", "p90_gap_s", "n_samples"]].to_string(index=False))

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F

# ============================================================
# MEASURE CO-ACTIVITY BETWEEN TAGS
# ============================================================
# Build a per-bucket presence matrix (1 = tag had data in bucket, 0 = not)
# Then compute Jaccard similarity: |A∩B| / |A∪B|
# Tags with high Jaccard similarity are active at the same time
# ============================================================

BUCKET_S = 10  # small bucket to resolve activity phases

df_presence = (
    df_raw_with_lift
    .filter(F.col("tagname").isin(tagnames))
    .withColumn("ts_unix", F.unix_timestamp(F.col("timestamp_utc").cast("timestamp")))
    .withColumn("bucket", (F.floor(F.col("ts_unix") / BUCKET_S) * BUCKET_S).cast("long"))
    .withColumn(
        "lift_bucket",
        F.concat_ws("_", F.col("lift_id").cast("string"), F.col("bucket").cast("string"))
    )
    .select("lift_bucket", "tagname")
    .distinct()
)

# Convert to wide presence matrix
print("Building presence matrix...")
presence_pdf = (
    df_presence
    .groupBy("lift_bucket")
    .pivot("tagname")
    .agg(F.lit(1))
    .toPandas()
    .set_index("lift_bucket")
    .fillna(0)
    .astype(int)
)

# Clean column names
presence_pdf.columns = [c.replace("ImportS7Device.", "") for c in presence_pdf.columns]

print(f"Presence matrix: {presence_pdf.shape[0]} buckets x {presence_pdf.shape[1]} tags")
print(f"Activity rate per tag (% of buckets where tag was active):")
activity_rate = (presence_pdf.mean() * 100).round(1).sort_values(ascending=False)
print(activity_rate.to_string())

# ============================================================
# JACCARD SIMILARITY MATRIX
# ============================================================
# Jaccard(A,B) = |A∩B| / |A∪B|
# High value = A and B are active together
# ============================================================

P = presence_pdf.values  # (n_buckets, n_tags)
cols = presence_pdf.columns.tolist()

# Intersection = dot product (both=1)
inter = P.T @ P
# Union = sum - intersection
col_sums = P.sum(axis=0)
union = col_sums[:, None] + col_sums[None, :] - inter

with np.errstate(divide="ignore", invalid="ignore"):
    jaccard = np.where(union > 0, inter / union, 0.0)

jaccard_df = pd.DataFrame(jaccard, index=cols, columns=cols)

print("Jaccard similarity computed")

# ============================================================
# HIERARCHICAL CLUSTERING ON (1 - JACCARD)
# ============================================================

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

dist = 1 - jaccard_df.values
np.fill_diagonal(dist, 0)
dist = np.clip(dist, 0, 1)

# Convert to condensed form
dist_condensed = squareform(dist, checks=False)

# Linkage
Z = linkage(dist_condensed, method="average")

# Cut tree at threshold — tune this:
#   0.3 = very tight groups (only tags that are nearly always co-active)
#   0.5 = medium
#   0.7 = loose
THRESHOLD = 0.5
cluster_ids = fcluster(Z, t=THRESHOLD, criterion="distance")

cluster_df = pd.DataFrame({
    "tag": cols,
    "activity_rate_%": activity_rate.values,
    "cluster": cluster_ids,
}).sort_values(["cluster", "activity_rate_%"], ascending=[True, False])

print(f"\n=== {cluster_df['cluster'].nunique()} clusters found at threshold={THRESHOLD} ===\n")
for cid, grp in cluster_df.groupby("cluster"):
    print(f"\n--- Cluster {cid}  ({len(grp)} tags, mean activity {grp['activity_rate_%'].mean():.1f}%) ---")
    for _, row in grp.iterrows():
        print(f"   {row['tag']:<55s} {row['activity_rate_%']:>5.1f}%")

        # ============================================================
# OPTIONAL: INSPECT THE JACCARD MATRIX
# ============================================================
# Print top-5 most similar tags for each tag

print("\n=== Top co-active partners per tag ===")
for tag in cols:
    top = jaccard_df[tag].drop(tag).sort_values(ascending=False).head(5)
    print(f"\n{tag}  (activity={activity_rate[tag]:.1f}%)")
    for partner, j in top.items():
        print(f"   {j:.2f}  {partner}")

# COMMAND ----------

# ============================================================
# ACTIVITY-BASED TAG GROUPING - FULL PIPELINE + TEST
# ============================================================
from pyspark.sql import functions as F
from pyspark.sql.functions import when
from itertools import chain
import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 220)

TIMESTAMP_COL = "timestamp_utc"

# ============================================================
# 1) ACTIVITY-BASED GROUPS
# ============================================================
# Move the slow-power / slow-voltage tags out of always_on
# Their true update behavior is delta-logged on big changes → needs large bucket

tagnames_always_on = [
    "ImportS7Device.CBM.LuffingWinch_Moment",
    "ImportS7Device.CBM.MainHoistWinch_Moment",
    "ImportS7Device.CBM.SlewBearing_Moment",
    "ImportS7Device.General.BoomLoad",
    "ImportS7Device.General.HoistLoadPctSWL",
    "ImportS7Device.General.PedestalMoment",
    "ImportS7Device.Drives.MainRectifier.DCLinkVoltage",  # gap=30s, keep
    "ImportS7Device.SafetySystems.AOPS RopeForce",
    "ImportS7Device.General.WindSpeed",
]

# NEW group: slow electrical signals that delta-log on large changes
tagnames_electrical_slow = [
    "ImportS7Device.Drives.MainRectifier.Power",
    "ImportS7Device.Drives.EmergRectifier.Power",
    "ImportS7Device.Drives.MainRectifier.SupplyVoltage",
    "ImportS7Device.Drives.EmergRectifier.SupplyVoltage",
    "ImportS7Device.Drives.EmergRectifier.DCLinkVoltage",
    "ImportS7Device.Energy.PACA.TotalActivePower",
    "ImportS7Device.Energy.PACB.TotalActivePower",
]

# --- hyd_common: ONLY the fast, well-sampled ones
tagnames_hyd_common = [
    "ImportS7Device.Hyd.BrakeSystemPressure",
    "ImportS7Device.Hyd.BrakeAccumulatorPressure",
]

# --- NEW: vessel/balance signals — tagged as "slow vessel state"
tagnames_vessel_state = [
    "ImportS7Device.General.HeelAngle",
    "ImportS7Device.General.TrimAngle",
    "ImportS7Device.Drives.Hoist_Status.ActualTorque",  # really a "load proxy"
]

tagnames_hyd_slow = [
    "ImportS7Device.Hyd.HoistPrimaryPressure",
    "ImportS7Device.Hyd.HoistSecondaryPressure",
    "ImportS7Device.Hyd.BoomSecondaryPressure",
    "ImportS7Device.Hyd.BrakesEnablePressure",
]

taggroups = {
    "always_on":         tagnames_always_on,
    "electrical_slow":   tagnames_electrical_slow,   # NEW
    "hoist_active":      tagnames_hoist_active,
    "boom_active":       tagnames_boom_active,
    "slew_active":       tagnames_slew_active,
    "hyd_common":        tagnames_hyd_common,
    "hyd_slow":          tagnames_hyd_slow,          # NEW
    "thermal_drive":     tagnames_thermal_drive,
    "thermal_resistor":  tagnames_thermal_resistor,
    "thermal_motor":     tagnames_thermal_motor,
    "slew_power":        tagnames_slew_power,
}

bucket_sizes = {
    "always_on":         5,
    "electrical_slow":  60,    # was 5 → fix the 595s gap problem
    "hoist_active":      5,
    "boom_active":       5,
    "slew_active":       5,
    "hyd_common":       10,
    "hyd_slow":         60,    # was 10 → fix the 580s gap problem
    "thermal_drive":    30,
    "thermal_resistor": 120,
    "thermal_motor":    300,
    "slew_power":       60,
}

taggroups["vessel_state"]    = tagnames_vessel_state
bucket_sizes["vessel_state"] = 60   # match the slower update rate
bucket_sizes["hyd_common"] = 10   # keep as-is, both tags have 10s median gap
tagnames = list(chain(*taggroups.values()))

print(f"Total groups: {len(taggroups)}")
print(f"Total tags:   {len(tagnames)}")
for g, tags in taggroups.items():
    print(f"  {g:<20s} bucket={bucket_sizes[g]:>4d}s  n={len(tags)}")

    # ============================================================
# 2) TAG GROUP MAPPING
# ============================================================

tag_group_expr = (
    when(F.col("tagname").isin(tagnames_always_on), "always_on")
    .when(F.col("tagname").isin(tagnames_hoist_active), "hoist_active")
    .when(F.col("tagname").isin(tagnames_boom_active), "boom_active")
    .when(F.col("tagname").isin(tagnames_slew_active), "slew_active")
    .when(F.col("tagname").isin(tagnames_hyd_common), "hyd_common")
    .when(F.col("tagname").isin(tagnames_thermal_drive), "thermal_drive")
    .when(F.col("tagname").isin(tagnames_thermal_resistor), "thermal_resistor")
    .when(F.col("tagname").isin(tagnames_thermal_motor), "thermal_motor")
    .when(F.col("tagname").isin(tagnames_slew_power), "slew_power")
    .otherwise("other")
)

# ============================================================
# 3) BUCKETING
# ============================================================

df_raw = (
    df_raw_with_lift
    .filter(F.col("tagname").isin(tagnames))
    .withColumn("tag_group", tag_group_expr)
    .withColumn("timestamp_utc", F.col("timestamp_utc").cast("timestamp"))
)

bucket_map_expr = F.create_map([F.lit(x) for x in chain(*bucket_sizes.items())])

df_raw_buckets = (
    df_raw
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

# ============================================================
# 4) PIVOT EACH GROUP
# ============================================================

df_pivoted_dict = {}
for group, tags in taggroups.items():
    df_group = df_raw_buckets.filter(F.col("tag_group") == group)
    if df_group.limit(1).count() > 0:
        df_pivoted_dict[group] = (
            df_group.groupBy("lift_group_bucket")
            .pivot("tagname", tags)
            .agg(F.mean("value"))
        )
        print(f"Pivoted {group:<20s} ({len(tags)} tags)")

# ============================================================
# 5) REMOVE lift_ids FROM df_liftlog_5
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

print(f"Removed {len(lift_ids_to_remove)} lift_ids (movementtype=5)")

# COMMAND ----------

# ============================================================
# 6) TEST GROUPS - TWO VIEWS OF "MISSINGNESS"
# ============================================================
# View A: ALL buckets per lift (where any group data exists).
#         NaN = "this subsystem was not in use"
# View B: Only buckets where at least one tag in this group had data.
#         NaN = "this tag was not logging even though subsystem was active"
# ============================================================

results = []

for group, df in df_pivoted_dict.items():
    print(f"\n=== Testing group: {group}  (bucket={bucket_sizes[group]}s) ===")
    pdf = df.toPandas().set_index("lift_group_bucket")
    pdf.columns = [c.replace("ImportS7Device.", "") for c in pdf.columns]

    n_total = len(pdf)

    # --- View A: all buckets
    miss_A_before = pdf.isna().mean() * 100
    miss_A_after  = pdf.ffill().isna().mean() * 100

    # --- View B: only active buckets (at least one tag has data)
    pdf_active = pdf.dropna(how="all")
    n_active = len(pdf_active)
    if n_active > 0:
        miss_B_before = pdf_active.isna().mean() * 100
        miss_B_after  = pdf_active.ffill().isna().mean() * 100
    else:
        miss_B_before = pd.Series(0.0, index=pdf.columns)
        miss_B_after  = pd.Series(0.0, index=pdf.columns)

    active_pct = 100 * n_active / n_total if n_total else 0
    print(f"  Total buckets:  {n_total}")
    print(f"  Active buckets: {n_active}  ({active_pct:.1f}% of total)")

    for col in pdf.columns:
        results.append({
            "Group":              group,
            "Bucket_s":           bucket_sizes[group],
            "Column":             col,
            "TotalBuckets":       n_total,
            "ActiveBuckets":      n_active,
            "ActivePct":          round(active_pct, 1),
            "A_MissBefore_%":     round(miss_A_before[col], 2),
            "A_MissAfterFFill_%": round(miss_A_after[col], 2),
            "B_MissBefore_%":     round(miss_B_before[col], 2),
            "B_MissAfterFFill_%": round(miss_B_after[col], 2),
        })

results_df = pd.DataFrame(results).sort_values(
    ["Group", "B_MissAfterFFill_%"]
).reset_index(drop=True)

# ============================================================
# 7) PRINT RESULTS
# ============================================================

print("\n\n========================= FULL RESULTS =========================")
print(results_df.to_string(index=False))

# ============================================================
# 8) SUMMARY PER GROUP (how well each group performs)
# ============================================================

group_summary = (
    results_df
    .groupby("Group")
    .agg(
        n_tags=("Column", "count"),
        bucket_s=("Bucket_s", "first"),
        total_buckets=("TotalBuckets", "first"),
        active_buckets=("ActiveBuckets", "first"),
        active_pct=("ActivePct", "first"),
        mean_miss_A_before=("A_MissBefore_%", "mean"),
        mean_miss_A_after=("A_MissAfterFFill_%", "mean"),
        mean_miss_B_before=("B_MissBefore_%", "mean"),
        mean_miss_B_after=("B_MissAfterFFill_%", "mean"),
        worst_tag_B_after=("B_MissAfterFFill_%", "max"),
    )
    .round(2)
    .reset_index()
)

print("\n\n========================= PER-GROUP SUMMARY =========================")
print(group_summary.to_string(index=False))

# ============================================================
# 9) VERDICT - which groups are "healthy"
# ============================================================

print("\n\n========================= VERDICT =========================")
for _, row in group_summary.iterrows():
    verdict = "✅ GOOD" if row["mean_miss_B_after"] < 10 else \
              "⚠️  OK"   if row["mean_miss_B_after"] < 25 else \
              "❌ BAD"
    print(f"  {verdict}  {row['Group']:<18s}  "
          f"B_miss_after={row['mean_miss_B_after']:>5.1f}%  "
          f"worst_tag={row['worst_tag_B_after']:>5.1f}%  "
          f"active={row['active_pct']:>5.1f}%")

# ============================================================
# 10) TSV for Excel
# ============================================================

print("\n\n--- TSV (Excel paste) ---\n")
print(results_df.to_csv(sep="\t", index=False))

# COMMAND ----------

import pandas as pd
import numpy as np

gap_stats = []

for group, df in df_pivoted_dict.items():
    pdf = df.toPandas()
    pdf[["lift_id", "bucket_unix"]] = pdf["lift_group_bucket"].str.rsplit("_", n=1, expand=True)
    pdf["bucket_unix"] = pdf["bucket_unix"].astype(np.int64)
    pdf = pdf.sort_values(["lift_id", "bucket_unix"]).reset_index(drop=True)

    bucket_s = bucket_sizes[group]
    data_cols = [c for c in pdf.columns
                 if c not in ("lift_group_bucket", "lift_id", "bucket_unix")]

    for col in data_cols:
        col_clean = col.replace("ImportS7Device.", "")
        gap_lengths = []

        for lift_id, sub in pdf.groupby("lift_id"):
            vals = sub[col].values
            # Find runs of NaN
            is_nan = np.isnan(vals)
            if not is_nan.any():
                continue
            # Run-length encode
            diffs = np.diff(np.concatenate(([0], is_nan.astype(int), [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            run_lengths = ends - starts
            # Exclude leading/trailing NaN gaps (can't interpolate those anyway)
            if len(run_lengths) > 0:
                # Check if first run starts at 0 (leading)
                if starts[0] == 0:
                    run_lengths = run_lengths[1:]
                    starts = starts[1:]
                # Check if last run ends at end (trailing)
                if len(run_lengths) > 0 and ends[-1] == len(vals):
                    run_lengths = run_lengths[:-1]

            gap_lengths.extend(run_lengths.tolist())

        if gap_lengths:
            gap_array = np.array(gap_lengths)
            gap_seconds = gap_array * bucket_s
            gap_stats.append({
                "Group": group,
                "BucketSec": bucket_s,
                "Column": col_clean,
                "NumGaps": len(gap_array),
                "MedianGap_buckets": int(np.median(gap_array)),
                "MedianGap_s": int(np.median(gap_seconds)),
                "P90Gap_s": int(np.percentile(gap_seconds, 90)),
                "P99Gap_s": int(np.percentile(gap_seconds, 99)),
                "MaxGap_s": int(gap_seconds.max()),
                "MeanGap_s": round(float(gap_seconds.mean()), 1),
            })
        else:
            gap_stats.append({
                "Group": group,
                "BucketSec": bucket_s,
                "Column": col_clean,
                "NumGaps": 0,
                "MedianGap_buckets": 0,
                "MedianGap_s": 0,
                "P90Gap_s": 0,
                "P99Gap_s": 0,
                "MaxGap_s": 0,
                "MeanGap_s": 0,
            })

gap_df = pd.DataFrame(gap_stats).sort_values(["Group", "P90Gap_s"], ascending=[True, False])

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 220)
print(gap_df.to_string(index=False))

# COMMAND ----------

max_gap_buckets = {
    "always_on":        17,   # 85 s  (cap WindSpeed's rare 220 s gap)
    "hoist_active":     17,   # 85 s
    "boom_active":      15,   # 75 s
    "slew_active":      26,   # 130 s (covers Encoders.Slew_Raw P99=127)
    "hyd_common":        2,   # 20 s
    "thermal_drive":     9,   # 270 s — won't fill the very long 510 s gaps
    "thermal_resistor":  3,   # 360 s
    "thermal_motor":     1,   # 300 s (one bucket only)
    "slew_power":        4,   # 240 s
}
import numpy as np
import pandas as pd

def weighted_fill_series(values, max_gap=10):
    """Distance-weighted temporal interpolation (left + right nearest neighbors)."""
    filled = values.astype(float).copy()
    n = len(filled)
    for i in range(n):
        if np.isnan(filled[i]):
            neighbors, weights = [], []
            for d in range(1, max_gap + 1):
                j = i - d
                if j >= 0 and not np.isnan(filled[j]):
                    neighbors.append(filled[j]); weights.append(1.0 / d); break
            for d in range(1, max_gap + 1):
                j = i + d
                if j < n and not np.isnan(filled[j]):
                    neighbors.append(filled[j]); weights.append(1.0 / d); break
            if weights:
                filled[i] = np.dot(neighbors, weights) / np.sum(weights)
    return filled


# ------------------------------------------------------------
# Apply weighted-fill per group, per lift, with group-specific max_gap
# ------------------------------------------------------------

df_filled_dict = {}

for group, df in df_pivoted_dict.items():
    max_gap = max_gap_buckets[group]
    pdf = df.toPandas()

    # split lift_id / bucket_unix
    pdf[["lift_id", "bucket_unix"]] = pdf["lift_group_bucket"].str.rsplit("_", n=1, expand=True)
    pdf["bucket_unix"] = pdf["bucket_unix"].astype(np.int64)
    pdf = pdf.sort_values(["lift_id", "bucket_unix"]).reset_index(drop=True)

    data_cols = [c for c in pdf.columns
                 if c not in ("lift_group_bucket", "lift_id", "bucket_unix")]

    # apply weighted_fill_series within each lift
    def fill_block(block):
        for col in data_cols:
            block[col] = weighted_fill_series(block[col].values, max_gap=max_gap)
        return block

    pdf = pdf.groupby("lift_id", group_keys=False).apply(fill_block)

    # report residual missingness
    miss = pdf[data_cols].isna().mean() * 100
    print(f"\n--- {group} (bucket={bucket_sizes[group]}s, max_gap={max_gap}) ---")
    for c in data_cols:
        print(f"   residual NaN {c.replace('ImportS7Device.', ''):<50s} {miss[c]:.2f}%")

    df_filled_dict[group] = pdf