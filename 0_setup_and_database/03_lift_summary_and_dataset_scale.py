# Databricks notebook source

df_raw_with_lift = spark.table("default.df_all_raw_liftid")
df_alarms_with_lift_fit = spark.table("default.df_alarms_fit")
df_liftlog = spark.table("default.all_liftlog")
df_alarms_with_lift = spark.table("default.df_alarms")




# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------
# 1. Basic dataset scale
# -----------------------------

unique_movementid_count = df_liftlog.select("movementid").distinct().count()

unique_liftid_with_sensor_data = df_raw_with_lift.select("lift_id").distinct().count()

unique_tagname_count = df_raw_with_lift.select("tagname").distinct().count()

raw_with_lift_row_count = df_raw_with_lift.count()

summary_rows = [
    ("Unique movementid in lift log", unique_movementid_count),
    ("Unique lift_id with aligned sensor data", unique_liftid_with_sensor_data),
    ("Unique sensor tags in aligned raw data", unique_tagname_count),
    ("Raw sensor observations aligned to lifts", raw_with_lift_row_count),
]

summary_df = spark.createDataFrame(summary_rows, ["Metric", "Value"])
display(summary_df)


# -----------------------------
# 2. Records per lift
# -----------------------------

liftid_value_counts = df_raw_with_lift.groupBy("lift_id").count()

lift_stats = liftid_value_counts.agg(
    F.count("*").alias("n_lifts_with_sensor_data"),
    F.min("count").alias("min_values_per_lift"),
    F.expr("percentile_approx(count, 0.50)").alias("median_values_per_lift"),
    F.avg("count").alias("mean_values_per_lift"),
    F.expr("percentile_approx(count, 0.90)").alias("p90_values_per_lift"),
    F.expr("percentile_approx(count, 0.95)").alias("p95_values_per_lift"),
    F.max("count").alias("max_values_per_lift"),
    F.sum("count").alias("total_values")
)

display(lift_stats)


# -----------------------------
# 3. Movement type distribution
# Use deduplicated lift log
# -----------------------------

df_liftlog_dedup = df_liftlog.dropDuplicates(["movementid"])

movementtype_counts = (
    df_liftlog_dedup
    .groupBy("movementtype")
    .count()
)

total_movements = df_liftlog_dedup.count()

movementtype_counts = (
    movementtype_counts
    .withColumn("share", F.col("count") / F.lit(total_movements))
    .orderBy("movementtype")
)

display(movementtype_counts)


# -----------------------------
# 4. Tagname counts
# -----------------------------

tagname_counts = (
    df_raw_with_lift
    .groupBy("tagname")
    .count()
    .orderBy(F.desc("count"))
)

display(tagname_counts)

parts = F.split(F.col("tagname"), "\\.")

tagname_counts_with_subsystem = (
    tagname_counts
    .withColumn("source", parts.getItem(0))
    .withColumn("subsystem", parts.getItem(1))
)

subsystem_summary = (
    tagname_counts_with_subsystem
    .groupBy("source", "subsystem")
    .agg(
        F.count("*").alias("n_tags"),
        F.sum("count").alias("n_observations")
    )
    .orderBy(F.desc("n_observations"))
)

display(subsystem_summary)

# COMMAND ----------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# OUT_DIR = "figures"
# os.makedirs(OUT_DIR, exist_ok=True)

tag_pdf = tagname_counts_with_subsystem.toPandas()
tag_pdf = tag_pdf.sort_values("count", ascending=False).reset_index(drop=True)

def shorten_tag(tag):
    tag = tag.replace("ImportS7Device.", "")
    tag = tag.replace("SharingOPCUA.", "")
    parts = tag.split(".")
    
    if len(parts) <= 3:
        return tag
    
    # Keep subsystem and last two levels
    return parts[0] + "." + parts[-2] + "." + parts[-1]

tag_pdf["short_tag"] = tag_pdf["tagname"].apply(shorten_tag)

top_n = 30

top_tags = (
    tag_pdf
    .head(top_n)
    .sort_values("count", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(top_tags["short_tag"], top_tags["count"])
ax.set_xscale("log")
ax.set_xlabel("Number of raw observations per tag, log scale")
ax.set_ylabel("Sensor tag")
ax.set_title("Top 30 sensor tags by observation count")

fig.tight_layout()
# fig.savefig(f"{OUT_DIR}/tag_count_top30_log.pdf", bbox_inches="tight")
plt.show()

bins = [0, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, np.inf]
labels = ["<1k", "1k-10k", "10k-100k", "100k-1M", "1M-10M", ">10M"]

tag_pdf["count_bin"] = pd.cut(
    tag_pdf["count"],
    bins=bins,
    labels=labels,
    right=False
)

bucket_pdf = (
    tag_pdf
    .groupby("count_bin", observed=False)
    .size()
    .reindex(labels)
    .reset_index(name="n_tags")
)

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.bar(bucket_pdf["count_bin"].astype(str), bucket_pdf["n_tags"])
ax.set_xlabel("Observation-count range")
ax.set_ylabel("Number of tags")
ax.set_title("Sensor tags by observation-count magnitude")

fig.tight_layout()
# fig.savefig(f"{OUT_DIR}/tag_count_magnitude_bins.pdf", bbox_inches="tight")
plt.show()

# COMMAND ----------

pareto_pdf = tag_pdf.sort_values("count", ascending=False).reset_index(drop=True)
pareto_pdf["rank"] = np.arange(1, len(pareto_pdf) + 1)
pareto_pdf["cumulative_share"] = pareto_pdf["count"].cumsum() / pareto_pdf["count"].sum()

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(pareto_pdf["rank"], pareto_pdf["cumulative_share"])
ax.set_xscale("log")
ax.set_ylim(0, 1.02)
ax.set_xlabel("Tag rank by observation count, log scale")
ax.set_ylabel("Cumulative share of observations")
ax.set_title("Cumulative observation share by ranked sensor tags")
ax.axhline(0.8, linestyle="--", linewidth=1)

fig.tight_layout()
# fig.savefig(f"{OUT_DIR}/tag_count_pareto.pdf", bbox_inches="tight")
plt.show()

# COMMAND ----------

subsystem_pdf = subsystem_summary.toPandas()

subsystem_pdf["label"] = (
    subsystem_pdf["source"] + "." + subsystem_pdf["subsystem"]
)

subsystem_obs_pdf = subsystem_pdf.sort_values("n_observations", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(subsystem_obs_pdf["label"], subsystem_obs_pdf["n_observations"])
ax.set_xscale("log")
ax.set_xlabel("Number of raw observations, log scale")
ax.set_ylabel("Subsystem")
ax.set_title("Observation volume by tag subsystem")

fig.tight_layout()
# fig.savefig(f"{OUT_DIR}/subsystem_observation_volume_log.pdf", bbox_inches="tight")
plt.show()

# COMMAND ----------

subsystem_tags_pdf = subsystem_pdf.sort_values("n_tags", ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(subsystem_tags_pdf["label"], subsystem_tags_pdf["n_tags"])
ax.set_xlabel("Number of unique tags")
ax.set_ylabel("Subsystem")
ax.set_title("Number of unique sensor tags by subsystem")

fig.tight_layout()
# fig.savefig(f"{OUT_DIR}/subsystem_unique_tags.pdf", bbox_inches="tight")
plt.show()

# COMMAND ----------

# ============================================================
# Export plot values for Overleaf / PGFPlots
# Requires:
#   df_liftlog
#   df_raw_with_lift
#
# Expected columns:
#   df_liftlog: movementid, movementtype
#   df_raw_with_lift: lift_id, tagname
# ============================================================

from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import os
import math
import textwrap

# ------------------------------------------------------------
# 0. Output folder
# ------------------------------------------------------------

# In Databricks, this will save files under:
# dbfs:/FileStore/thesis_overleaf_values/
# You can download them from:
# /files/thesis_overleaf_values/<filename>
try:
    dbutils.fs.mkdirs("dbfs:/FileStore/thesis_overleaf_values")
    OUT_DIR = "/dbfs/FileStore/thesis_overleaf_values"
except Exception:
    OUT_DIR = "thesis_overleaf_values"

os.makedirs(OUT_DIR, exist_ok=True)

PRINT_FULL_OUTPUT_TO_CONSOLE = True
PRINT_FULL_PARETO_TO_CONSOLE = True  # Set False if Databricks output becomes too long


# ------------------------------------------------------------
# 1. Helper functions
# ------------------------------------------------------------

def tex_escape(text):
    """Escape text so it can be used safely in LaTeX labels."""
    if text is None:
        return ""
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def shorten_tag(tag):
    """Shorten full tag names for thesis plots."""
    tag = str(tag)
    tag = tag.replace("ImportS7Device.", "")
    tag = tag.replace("SharingOPCUA.", "")
    return tag


def subsystem_label(source, subsystem):
    """Make subsystem labels readable."""
    if source == "ImportS7Device":
        return str(subsystem)
    return f"{source}.{subsystem}"


def write_text(filename, text):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote: {path}")


def print_block(title, text, print_full=True):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if print_full:
        print(text)


def dataframe_to_latex_rows(df, columns):
    """Return simple LaTeX table rows from selected columns."""
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(tex_escape(value))
        rows.append(" & ".join(values) + r" \\")
    return "\n".join(rows)


def coordinates_from_df(df, x_col, y_col):
    """Return PGFPlots coordinates."""
    return "\n".join(
        f"({row[x_col]},{row[y_col]})"
        for _, row in df.iterrows()
    )


def ytick_list(labels):
    """Return LaTeX yticklabels list."""
    return ",\n".join("{" + tex_escape(label) + "}" for label in labels)


# ------------------------------------------------------------
# 2. Basic dataset scale
# ------------------------------------------------------------

movement_col = "movementtype" if "movementtype" in df_liftlog.columns else "movement_type"

unique_movementid_count = df_liftlog.select("movementid").distinct().count()
unique_liftid_with_sensor_data = df_raw_with_lift.select("lift_id").distinct().count()
unique_tagname_count = df_raw_with_lift.select("tagname").distinct().count()
raw_with_lift_row_count = df_raw_with_lift.count()

liftid_value_counts = df_raw_with_lift.groupBy("lift_id").count()

lift_stats_spark = liftid_value_counts.agg(
    F.count("*").alias("n_lifts_with_sensor_data"),
    F.min("count").alias("min_values_per_lift"),
    F.expr("percentile_approx(count, 0.50, 10000)").alias("median_values_per_lift"),
    F.avg("count").alias("mean_values_per_lift"),
    F.expr("percentile_approx(count, 0.90, 10000)").alias("p90_values_per_lift"),
    F.expr("percentile_approx(count, 0.95, 10000)").alias("p95_values_per_lift"),
    F.max("count").alias("max_values_per_lift"),
    F.sum("count").alias("total_values")
)

lift_stats = lift_stats_spark.toPandas().iloc[0].to_dict()

dataset_scale_rows = [
    ("Unique movement intervals in lift log", unique_movementid_count),
    ("Unique lift intervals with aligned sensor data", unique_liftid_with_sensor_data),
    ("Unique sensor tags in aligned raw data", unique_tagname_count),
    ("Raw sensor observations aligned to lifts", raw_with_lift_row_count),
    ("Median observations per lift", int(lift_stats["median_values_per_lift"])),
    ("Mean observations per lift", round(float(lift_stats["mean_values_per_lift"]), 2)),
    ("95th percentile observations per lift", int(lift_stats["p95_values_per_lift"])),
    ("Maximum observations in one lift", int(lift_stats["max_values_per_lift"])),
]

dataset_scale_pdf = pd.DataFrame(dataset_scale_rows, columns=["Metric", "Value"])

dataset_scale_latex = r"""\begin{table}[htbp]
    \centering
    \caption{Scale of the lift-aligned raw sensor dataset.}
    \label{tab:dataset-scale}
    \begin{tabular}{lr}
        \toprule
        Metric & Value \\
        \midrule
""" + "\n".join(
    f"        {tex_escape(metric)} & {value} \\\\"
    for metric, value in dataset_scale_rows
) + r"""
        \bottomrule
    \end{tabular}
\end{table}
"""

write_text("tab_dataset_scale.tex", dataset_scale_latex)

print_block(
    "DATASET SCALE VALUES",
    dataset_scale_pdf.to_string(index=False),
    PRINT_FULL_OUTPUT_TO_CONSOLE
)


# ------------------------------------------------------------
# 3. Movement type distribution
# ------------------------------------------------------------

df_liftlog_dedup = df_liftlog.dropDuplicates(["movementid"])

movement_raw = (
    df_liftlog
    .groupBy(movement_col)
    .count()
    .withColumnRenamed("count", "raw_count")
)

movement_dedup = (
    df_liftlog_dedup
    .groupBy(movement_col)
    .count()
    .withColumnRenamed("count", "deduplicated_count")
)

movement_pdf = (
    movement_raw
    .join(movement_dedup, on=movement_col, how="outer")
    .fillna(0)
    .orderBy(movement_col)
    .toPandas()
)

movement_pdf["raw_count"] = movement_pdf["raw_count"].astype(int)
movement_pdf["deduplicated_count"] = movement_pdf["deduplicated_count"].astype(int)
movement_pdf["share"] = movement_pdf["deduplicated_count"] / movement_pdf["deduplicated_count"].sum()
movement_pdf["movementtype"] = movement_pdf[movement_col].astype(str)

movement_tsv = movement_pdf[["movementtype", "raw_count", "deduplicated_count", "share"]].to_csv(
    sep="\t", index=False
)
write_text("movementtype_distribution.tsv", movement_tsv)

movement_coords_dedup = "\n".join(
    f"({row['movementtype']},{row['deduplicated_count']})"
    for _, row in movement_pdf.iterrows()
)

movement_coords_raw = "\n".join(
    f"({row['movementtype']},{row['raw_count']})"
    for _, row in movement_pdf.iterrows()
)

movement_tex = r"""\begin{figure}[htbp]
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        width=0.82\linewidth,
        height=0.42\linewidth,
        ybar,
        bar width=8pt,
        xlabel={Movement type},
        ylabel={Number of lift intervals},
        symbolic x coords={1,2,3,4,5},
        xtick=data,
        ymin=0,
        legend style={at={(0.5,-0.18)}, anchor=north, legend columns=2},
        grid=major
    ]
        \addplot coordinates {
""" + movement_coords_raw + r"""
        };
        \addplot coordinates {
""" + movement_coords_dedup + r"""
        };
        \legend{Raw count, Deduplicated count}
    \end{axis}
    \end{tikzpicture}
    \caption{Distribution of movement types before and after duplicate lift entries were removed.}
    \label{fig:movementtype-distribution}
\end{figure}
"""

write_text("fig_movementtype_distribution.tex", movement_tex)

print_block(
    "MOVEMENT TYPE VALUES",
    movement_pdf[["movementtype", "raw_count", "deduplicated_count", "share"]].to_string(index=False),
    PRINT_FULL_OUTPUT_TO_CONSOLE
)


# ------------------------------------------------------------
# 4. Tagname counts
# ------------------------------------------------------------

tagname_counts = (
    df_raw_with_lift
    .groupBy("tagname")
    .count()
    .orderBy(F.desc("count"))
)

parts = F.split(F.col("tagname"), "\\.")

tagname_counts_with_subsystem = (
    tagname_counts
    .withColumn("source", parts.getItem(0))
    .withColumn("subsystem", parts.getItem(1))
)

tag_pdf = tagname_counts_with_subsystem.toPandas()
tag_pdf = tag_pdf.sort_values("count", ascending=False).reset_index(drop=True)
tag_pdf["count"] = tag_pdf["count"].astype(int)
tag_pdf["rank"] = np.arange(1, len(tag_pdf) + 1)
tag_pdf["short_tag"] = tag_pdf["tagname"].apply(shorten_tag)
tag_pdf["share"] = tag_pdf["count"] / tag_pdf["count"].sum()
tag_pdf["cumulative_share"] = tag_pdf["count"].cumsum() / tag_pdf["count"].sum()

tag_all_tsv = tag_pdf[
    ["rank", "tagname", "short_tag", "source", "subsystem", "count", "share", "cumulative_share"]
].to_csv(sep="\t", index=False)

write_text("all_tag_counts.tsv", tag_all_tsv)


# ------------------------------------------------------------
# 5. Top 30 tag-count plot values
# ------------------------------------------------------------

top_n = 30
top_tags = tag_pdf.head(top_n).copy()
top_tags["plot_y"] = np.arange(1, len(top_tags) + 1)

top_tags_tsv = top_tags[
    ["plot_y", "rank", "short_tag", "tagname", "count", "share", "cumulative_share"]
].to_csv(sep="\t", index=False)

write_text("top30_tag_counts.tsv", top_tags_tsv)

top30_coords = "\n".join(
    f"({row['count']},{row['plot_y']})"
    for _, row in top_tags.iterrows()
)

top30_yticks = ",".join(str(i) for i in top_tags["plot_y"])
top30_yticklabels = ytick_list(top_tags["short_tag"])

# Dot plot is safer than a bar chart on a logarithmic x-axis.
# Bars have a zero baseline, and zero is not valid on a log scale.
top30_tex = r"""\begin{figure}[htbp]
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        width=\linewidth,
        height=0.78\textheight,
        xmode=log,
        xlabel={Number of raw observations per tag},
        ylabel={Sensor tag},
        y dir=reverse,
        ytick={""" + top30_yticks + r"""},
        yticklabels={
""" + top30_yticklabels + r"""
        },
        yticklabel style={font=\scriptsize, text width=0.38\linewidth, align=right},
        grid=both,
        xmin=1000000
    ]
        \addplot+[only marks, mark=*, mark size=1.8pt] coordinates {
""" + top30_coords + r"""
        };
    \end{axis}
    \end{tikzpicture}
    \caption{Top 30 sensor tags by number of raw observations in the lift-aligned dataset. A logarithmic x-axis is used because the tag counts span several orders of magnitude.}
    \label{fig:tag-count-top30}
\end{figure}
"""

write_text("fig_tag_count_top30.tex", top30_tex)

print_block(
    "TOP 30 TAG VALUES",
    top_tags[["rank", "short_tag", "count", "share", "cumulative_share"]].to_string(index=False),
    PRINT_FULL_OUTPUT_TO_CONSOLE
)


# ------------------------------------------------------------
# 6. Observation-count magnitude bins
# ------------------------------------------------------------

bins = [0, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, np.inf]
labels = ["<1k", "1k-10k", "10k-100k", "100k-1M", "1M-10M", ">10M"]
labels_tex = [r"$<1$k", "1k--10k", "10k--100k", "100k--1M", "1M--10M", r"$>10$M"]

tag_pdf["count_bin"] = pd.cut(
    tag_pdf["count"],
    bins=bins,
    labels=labels,
    right=False
)

bucket_pdf = (
    tag_pdf
    .groupby("count_bin", observed=False)
    .agg(
        n_tags=("tagname", "count"),
        n_observations=("count", "sum")
    )
    .reindex(labels)
    .reset_index()
)

bucket_pdf["plot_x"] = np.arange(1, len(bucket_pdf) + 1)
bucket_pdf["label_tex"] = labels_tex
bucket_pdf["tag_share"] = bucket_pdf["n_tags"] / bucket_pdf["n_tags"].sum()
bucket_pdf["observation_share"] = bucket_pdf["n_observations"] / bucket_pdf["n_observations"].sum()

bucket_tsv = bucket_pdf[
    ["plot_x", "count_bin", "label_tex", "n_tags", "n_observations", "tag_share", "observation_share"]
].to_csv(sep="\t", index=False)

write_text("tag_count_magnitude_bins.tsv", bucket_tsv)

bucket_coords = "\n".join(
    f"({int(row['plot_x'])},{int(row['n_tags'])})"
    for _, row in bucket_pdf.iterrows()
)

bucket_xticks = ",".join(str(int(x)) for x in bucket_pdf["plot_x"])
bucket_xticklabels = ",".join("{" + label + "}" for label in bucket_pdf["label_tex"])

bucket_tex = r"""\begin{figure}[htbp]
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        width=0.82\linewidth,
        height=0.45\linewidth,
        ybar,
        bar width=16pt,
        xlabel={Observation-count range},
        ylabel={Number of tags},
        xtick={""" + bucket_xticks + r"""},
        xticklabels={""" + bucket_xticklabels + r"""},
        ymin=0,
        grid=major
    ]
        \addplot coordinates {
""" + bucket_coords + r"""
        };
    \end{axis}
    \end{tikzpicture}
    \caption{Distribution of sensor tags by observation-count magnitude in the lift-aligned raw dataset.}
    \label{fig:tag-count-magnitude}
\end{figure}
"""

write_text("fig_tag_count_magnitude_bins.tex", bucket_tex)

print_block(
    "MAGNITUDE BIN VALUES",
    bucket_pdf[
        ["count_bin", "n_tags", "n_observations", "tag_share", "observation_share"]
    ].to_string(index=False),
    PRINT_FULL_OUTPUT_TO_CONSOLE
)


# ------------------------------------------------------------
# 7. Pareto curve values
# ------------------------------------------------------------

pareto_pdf = tag_pdf[["rank", "short_tag", "tagname", "count", "share", "cumulative_share"]].copy()

n80 = int(pareto_pdf.loc[pareto_pdf["cumulative_share"] >= 0.80, "rank"].iloc[0])
top6_share = pareto_pdf.head(6)["count"].sum() / pareto_pdf["count"].sum()
top10_share = pareto_pdf.head(10)["count"].sum() / pareto_pdf["count"].sum()
top30_share = pareto_pdf.head(30)["count"].sum() / pareto_pdf["count"].sum()

pareto_tsv = pareto_pdf.to_csv(sep="\t", index=False)
write_text("tag_count_pareto_all_values.tsv", pareto_tsv)

pareto_coords = "\n".join(
    f"({int(row['rank'])},{float(row['cumulative_share']):.8f})"
    for _, row in pareto_pdf.iterrows()
)

pareto_tex = r"""\begin{figure}[htbp]
    \centering
    \begin{tikzpicture}
    \begin{semilogxaxis}[
        width=0.86\linewidth,
        height=0.48\linewidth,
        xlabel={Tag rank by observation count},
        ylabel={Cumulative share of observations},
        ymin=0,
        ymax=1.02,
        xmin=1,
        xmax=""" + str(int(pareto_pdf["rank"].max())) + r""",
        grid=both
    ]
        \addplot+[mark=none, thick] coordinates {
""" + pareto_coords + r"""
        };

        \addplot[dashed, mark=none] coordinates {
            (1,0.8)
            (""" + str(int(pareto_pdf["rank"].max())) + r""",0.8)
        };

        \addplot[dashed, mark=none] coordinates {
            (""" + str(n80) + r""",0)
            (""" + str(n80) + r""",1)
        };

        \node[anchor=south east, rotate=90] at (axis cs:""" + str(n80) + r""",0.05)
        {80\% at """ + str(n80) + r""" tags};
    \end{semilogxaxis}
    \end{tikzpicture}
    \caption{Cumulative share of raw observations by ranked sensor tags.}
    \label{fig:tag-count-pareto}
\end{figure}
"""

write_text("fig_tag_count_pareto.tex", pareto_tex)

pareto_summary = f"""
Top 6 tags share:  {top6_share:.6f}  ({top6_share:.2%})
Top 10 tags share: {top10_share:.6f}  ({top10_share:.2%})
Top 30 tags share: {top30_share:.6f}  ({top30_share:.2%})
80% of observations reached at tag rank: {n80}
"""

print_block("PARETO SUMMARY VALUES", pareto_summary, PRINT_FULL_OUTPUT_TO_CONSOLE)

if PRINT_FULL_PARETO_TO_CONSOLE:
    print_block(
        "FULL PARETO VALUES",
        pareto_pdf[["rank", "short_tag", "count", "share", "cumulative_share"]].to_string(index=False),
        True
    )


# ------------------------------------------------------------
# 8. Subsystem summaries
# ------------------------------------------------------------

subsystem_summary = (
    tagname_counts_with_subsystem
    .groupBy("source", "subsystem")
    .agg(
        F.count("*").alias("n_tags"),
        F.sum("count").alias("n_observations")
    )
    .orderBy(F.desc("n_observations"))
)

subsystem_pdf = subsystem_summary.toPandas()
subsystem_pdf["n_tags"] = subsystem_pdf["n_tags"].astype(int)
subsystem_pdf["n_observations"] = subsystem_pdf["n_observations"].astype(int)
subsystem_pdf["label"] = subsystem_pdf.apply(
    lambda r: subsystem_label(r["source"], r["subsystem"]),
    axis=1
)
subsystem_pdf["tag_share"] = subsystem_pdf["n_tags"] / subsystem_pdf["n_tags"].sum()
subsystem_pdf["observation_share"] = subsystem_pdf["n_observations"] / subsystem_pdf["n_observations"].sum()

subsystem_tsv = subsystem_pdf[
    ["source", "subsystem", "label", "n_tags", "n_observations", "tag_share", "observation_share"]
].to_csv(sep="\t", index=False)

write_text("subsystem_summary.tsv", subsystem_tsv)


# 8A. Observation volume by subsystem, log-scale dot plot
subsystem_obs = subsystem_pdf.sort_values("n_observations", ascending=False).reset_index(drop=True)
subsystem_obs["plot_y"] = np.arange(1, len(subsystem_obs) + 1)

subsystem_obs_coords = "\n".join(
    f"({int(row['n_observations'])},{int(row['plot_y'])})"
    for _, row in subsystem_obs.iterrows()
)

subsystem_obs_yticks = ",".join(str(int(y)) for y in subsystem_obs["plot_y"])
subsystem_obs_yticklabels = ytick_list(subsystem_obs["label"])

subsystem_obs_tex = r"""\begin{figure}[htbp]
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        width=0.9\linewidth,
        height=0.62\linewidth,
        xmode=log,
        xlabel={Number of raw observations},
        ylabel={Subsystem},
        y dir=reverse,
        ytick={""" + subsystem_obs_yticks + r"""},
        yticklabels={
""" + subsystem_obs_yticklabels + r"""
        },
        yticklabel style={font=\scriptsize, text width=0.28\linewidth, align=right},
        grid=both,
        xmin=10000
    ]
        \addplot+[only marks, mark=*, mark size=1.8pt] coordinates {
""" + subsystem_obs_coords + r"""
        };
    \end{axis}
    \end{tikzpicture}
    \caption{Observation volume by tag subsystem.}
    \label{fig:subsystem-observation-volume}
\end{figure}
"""

write_text("fig_subsystem_observation_volume.tex", subsystem_obs_tex)


# 8B. Number of unique tags by subsystem, normal horizontal bar chart
subsystem_tags = subsystem_pdf.sort_values("n_tags", ascending=False).reset_index(drop=True)
subsystem_tags["plot_y"] = np.arange(1, len(subsystem_tags) + 1)

subsystem_tags_coords = "\n".join(
    f"({int(row['n_tags'])},{int(row['plot_y'])})"
    for _, row in subsystem_tags.iterrows()
)

subsystem_tags_yticks = ",".join(str(int(y)) for y in subsystem_tags["plot_y"])
subsystem_tags_yticklabels = ytick_list(subsystem_tags["label"])

subsystem_tags_tex = r"""\begin{figure}[htbp]
    \centering
    \begin{tikzpicture}
    \begin{axis}[
        width=0.9\linewidth,
        height=0.62\linewidth,
        xbar,
        xlabel={Number of unique tags},
        ylabel={Subsystem},
        y dir=reverse,
        ytick={""" + subsystem_tags_yticks + r"""},
        yticklabels={
""" + subsystem_tags_yticklabels + r"""
        },
        yticklabel style={font=\scriptsize, text width=0.28\linewidth, align=right},
        xmin=0,
        grid=major
    ]
        \addplot coordinates {
""" + subsystem_tags_coords + r"""
        };
    \end{axis}
    \end{tikzpicture}
    \caption{Number of unique sensor tags by subsystem.}
    \label{fig:subsystem-unique-tags}
\end{figure}
"""

write_text("fig_subsystem_unique_tags.tex", subsystem_tags_tex)

print_block(
    "SUBSYSTEM SUMMARY VALUES",
    subsystem_pdf[
        ["label", "n_tags", "n_observations", "tag_share", "observation_share"]
    ].to_string(index=False),
    PRINT_FULL_OUTPUT_TO_CONSOLE
)


# ------------------------------------------------------------
# 9. Compact text summary for thesis writing
# ------------------------------------------------------------

summary_text = f"""
Dataset scale:
- Unique movement intervals in lift log: {unique_movementid_count}
- Unique lift intervals with aligned sensor data: {unique_liftid_with_sensor_data}
- Unique sensor tags: {unique_tagname_count}
- Raw sensor observations aligned to lifts: {raw_with_lift_row_count}
- Median observations per lift: {int(lift_stats["median_values_per_lift"])}
- Mean observations per lift: {float(lift_stats["mean_values_per_lift"]):.2f}
- P95 observations per lift: {int(lift_stats["p95_values_per_lift"])}
- Maximum observations in one lift: {int(lift_stats["max_values_per_lift"])}

Tag-count concentration:
- Top 6 tags share: {top6_share:.2%}
- Top 10 tags share: {top10_share:.2%}
- Top 30 tags share: {top30_share:.2%}
- 80% of observations reached at tag rank: {n80}
"""

write_text("summary_values.txt", summary_text)

print_block("COMPACT SUMMARY", summary_text, PRINT_FULL_OUTPUT_TO_CONSOLE)

print("\nDone.")
print(f"All Overleaf values and figure snippets were written to: {OUT_DIR}")

# COMMAND ----------

for filename in sorted(os.listdir(OUT_DIR)):
    if filename.endswith(".tex") or filename.endswith(".txt"):
        path = os.path.join(OUT_DIR, filename)
        print(f"\n{'='*80}\n{filename}\n{'='*80}")
        with open(path, "r", encoding="utf-8") as f:
            print(f.read())

# COMMAND ----------

# Analyze lift durations in df_liftlog

lift_duration_stats = (
    df_liftlog
    .select("movementid", "movementtype", "movementduration")
    .groupBy("movementtype")
    .agg(
        F.count("*").alias("n_lifts"),
        F.min("movementduration").alias("min_duration"),
        F.expr("percentile_approx(movementduration, 0.50)").alias("median_duration"),
        F.avg("movementduration").alias("mean_duration"),
        F.expr("percentile_approx(movementduration, 0.90)").alias("p90_duration"),
        F.expr("percentile_approx(movementduration, 0.95)").alias("p95_duration"),
        F.max("movementduration").alias("max_duration")
    )
    .orderBy("movementtype")
)

display(lift_duration_stats)