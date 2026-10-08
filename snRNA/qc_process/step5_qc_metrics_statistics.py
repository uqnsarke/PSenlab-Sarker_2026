#!/usr/bin/env python3
#===============================================================================
# QC metric distributions and statistics for snRNA-seq (Extended Data Fig. 1c)
#===============================================================================
# Description: Re-draws the RNA QC violin + box plots by age group and writes
#              the numbers reported in the figure legend and Source Data:
#                - per age group: n nuclei, n mice, minimum, lower whisker,
#                  25th percentile, median, mean, 75th percentile,
#                  upper whisker, maximum, n values beyond the whiskers
#                - per mouse: n nuclei and median of each metric
#                - age-group tests on per-mouse medians
#              The unit of study is the mouse (n = 8 per age group, 4 female +
#              4 male); nuclei from one mouse are not independent replicates,
#              so tests are run on one median per mouse.
#
# Input:       Final snRNA AnnData (only .obs is read) with columns
#              sample_name, age, sex, total_counts, n_genes_by_counts
# Output:      ED_Fig1c_replot.pdf / .png
#              ED_Fig1c_source_data_per_age.csv
#              ED_Fig1c_source_data_per_mouse.csv
#              ED_Fig1c_stats_per_mouse.csv
#              ED_Fig1c_source_data.xlsx (the three tables above)
#
# Box plots:   centre = median; box = 25th and 75th percentiles; lower whisker
#              = smallest value >= Q1 - 1.5 x IQR; upper whisker = largest value
#              <= Q3 + 1.5 x IQR; values beyond the whiskers are not drawn.
#
# Statistics:  Kruskal-Wallis (and one-way ANOVA as a check) across the five
#              age groups on per-mouse medians, sexes combined; Benjamini-
#              Hochberg correction across the two RNA metrics.
#              Effect size: rank eta-squared, eta2_H = (H - k + 1) / (N - k), with a
#              95% bootstrap CI (mice resampled within age groups).
#===============================================================================

import os
import logging
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global plot settings
mpl.rcParams["font.family"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
INPUT_FILE = "snRNA_WNN.h5ad"
OUTPUT_DIR = "qc_metrics"

# Columns in adata.obs
SAMPLE_COL = "sample_name"
AGE_COL = "age"
SEX_COL = "sex"

# QC metrics to summarise: obs column -> axis label
RNA_METRICS = {
    "total_counts": "n counts RNA",
    "n_genes_by_counts": "n genes by counts",
}

# Age group order and colors
AGE_ORDER = ["young", "mid-age", "old", "pre-geriatric", "geriatric"]
AGE_COLORS = {
    "young": "#1ABC9C",
    "mid-age": "#F1C40F",
    "old": "#C39BD3",
    "pre-geriatric": "#2980B9",
    "geriatric": "#E84393",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


#-------------------------------------------------------------------------------
# Helpers
#-------------------------------------------------------------------------------
def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def kruskal_effect(groups, n_boot=10000, seed=2026):
    """Kruskal-Wallis test across groups, with rank eta-squared and a bootstrap CI.

    Returns H, df, N, P, eta2_H, CI low, CI high.
    eta2_H = (H - k + 1) / (N - k), negative values set to 0.
    The 95% CI is the percentile interval from bootstrap resamples of mice
    within each age group (fixed seed, so the numbers are reproducible).
    """
    k = len(groups)
    n = sum(len(g) for g in groups)

    def eta2(h):
        return max(0.0, (h - k + 1) / (n - k))

    res = stats.kruskal(*groups)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        resampled = [rng.choice(g, size=len(g), replace=True) for g in groups]
        boot[b] = eta2(stats.kruskal(*resampled).statistic)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return res.statistic, k - 1, n, res.pvalue, eta2(res.statistic), lo, hi


#-------------------------------------------------------------------------------
# Load (only obs is read)
#-------------------------------------------------------------------------------
adata = ad.read_h5ad(INPUT_FILE, backed="r")
obs = adata.obs.copy()
adata.file.close()

df = (obs[[SAMPLE_COL, AGE_COL, SEX_COL, *RNA_METRICS]]
      .rename(columns={SAMPLE_COL: "sample", AGE_COL: "age", SEX_COL: "sex"}).copy())
df["age"] = pd.Categorical(df["age"].astype(str).str.replace("_", "-").str.lower(),
                           categories=AGE_ORDER, ordered=True)
df["sex"] = df["sex"].astype(str).str.lower()
df["sample"] = df["sample"].astype(str)
assert df["age"].notna().all(), "unrecognised age labels"

mice = df.drop_duplicates("sample")
logger.info(f"{len(df):,} nuclei, {df['sample'].nunique()} mice")
print(pd.crosstab(mice["age"], mice["sex"]))      # expect 4 female + 4 male per age group

#-------------------------------------------------------------------------------
# Plot: violin (density of all nuclei) with inner box plot
#-------------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(RNA_METRICS), figsize=(7, 3.6))
pos = np.arange(1, len(AGE_ORDER) + 1)
for ax, (metric, label) in zip(axes, RNA_METRICS.items()):
    data = [df.loc[df["age"] == age, metric].values for age in AGE_ORDER]
    parts = ax.violinplot(data, positions=pos, widths=0.9, showextrema=False)
    for body, age in zip(parts["bodies"], AGE_ORDER):
        body.set_facecolor(AGE_COLORS[age])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(1)
    ax.boxplot(data, positions=pos, widths=0.1, showfliers=False, whis=1.5, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="black", linewidth=0.7),
               medianprops=dict(color="black", linewidth=0.9),
               whiskerprops=dict(linewidth=0.7), capprops=dict(linewidth=0))
    ax.set_xticks(pos)
    ax.set_xticklabels(AGE_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(label, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.text(0.01, 0.98, "c", fontsize=18, va="top")
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/ED_Fig1c_replot.pdf", bbox_inches="tight")
fig.savefig(f"{OUTPUT_DIR}/ED_Fig1c_replot.png", bbox_inches="tight", dpi=200)
plt.close()
logger.info("Saved ED_Fig1c_replot")

#-------------------------------------------------------------------------------
# Box plot statistics per age group (all nuclei)
#-------------------------------------------------------------------------------
rows = []
for metric in RNA_METRICS:
    for age in AGE_ORDER:
        sub = df[df["age"] == age]
        v = sub[metric].values
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        rows.append(dict(
            metric=metric, age=age, n_nuclei=len(v), n_mice=sub["sample"].nunique(),
            minimum=v.min(), whisker_low=v[v >= q1 - 1.5 * iqr].min(),
            Q1_25th_percentile=q1, median=med, mean=v.mean(), Q3_75th_percentile=q3,
            whisker_high=v[v <= q3 + 1.5 * iqr].max(), maximum=v.max(),
            n_outliers_not_plotted=int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()),
        ))
exact = pd.DataFrame(rows)
exact.to_csv(f"{OUTPUT_DIR}/ED_Fig1c_source_data_per_age.csv", index=False)
print(exact.round(1).to_string(index=False))

#-------------------------------------------------------------------------------
# Per-mouse table (unit of study): n nuclei and median of each metric
#-------------------------------------------------------------------------------
per_mouse = df.groupby(["sample", "age", "sex"], observed=True).agg(
    n_nuclei=("sample", "size"),
    **{f"median_{m}": (m, "median") for m in RNA_METRICS},
).reset_index().sort_values(["age", "sex", "sample"])
per_mouse.to_csv(f"{OUTPUT_DIR}/ED_Fig1c_source_data_per_mouse.csv", index=False)
print(per_mouse.round(1).to_string(index=False))

#-------------------------------------------------------------------------------
# Age-group tests on per-mouse medians (sexes combined)
#-------------------------------------------------------------------------------
rows = []
for metric in RNA_METRICS:
    groups = [g[f"median_{metric}"].values
              for _, g in per_mouse.groupby("age", observed=True)]
    H, df_kw, n_mice, p, eta2, ci_lo, ci_hi = kruskal_effect(groups)
    an = stats.f_oneway(*groups)
    rows.append(dict(metric=metric, test="Kruskal-Wallis", N_mice=n_mice,
                     n_per_group=",".join(str(len(g)) for g in groups), df=df_kw,
                     kruskal_H=H, kruskal_p=p,
                     eta2_H=eta2, eta2_H_CI95_low=ci_lo, eta2_H_CI95_high=ci_hi,
                     anova_F=an.statistic, anova_df=f"{df_kw},{n_mice - len(groups)}",
                     anova_p=an.pvalue))
tests = pd.DataFrame(rows)
tests["kruskal_p_BH"] = bh(tests["kruskal_p"].values)     # across the two RNA metrics
tests["anova_p_BH"] = bh(tests["anova_p"].values)
tests.to_csv(f"{OUTPUT_DIR}/ED_Fig1c_stats_per_mouse.csv", index=False)
print(tests.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

#-------------------------------------------------------------------------------
# One Excel file for Source Data (skipped if openpyxl is not installed)
#-------------------------------------------------------------------------------
try:
    with pd.ExcelWriter(f"{OUTPUT_DIR}/ED_Fig1c_source_data.xlsx") as xw:
        exact.to_excel(xw, sheet_name="per_age_group", index=False)
        per_mouse.to_excel(xw, sheet_name="per_mouse", index=False)
        tests.to_excel(xw, sheet_name="tests", index=False)
        pd.DataFrame({"note": [
            "Unit of study: mouse (one median per mouse and metric); N_mice mice, n_per_group mice per age group.",
            "kruskal_H, df, kruskal_p: Kruskal-Wallis test across the five age groups (non-directional omnibus test, chi-square approximation with tie correction).",
            "kruskal_p_BH: Benjamini-Hochberg adjusted P, correction across the two metrics of this modality.",
            "eta2_H: rank eta-squared, (H - k + 1) / (N - k) with k = 5 age groups, negative values set to 0.",
            "eta2_H_CI95_low / eta2_H_CI95_high: 95% percentile interval from 10,000 bootstrap resamples of mice within age groups (seed 2026).",
            "anova_*: one-way ANOVA on the same per-mouse medians, shown as a check.",
        ]}).to_excel(xw, sheet_name="notes", index=False)
    logger.info("Wrote ED_Fig1c_source_data.xlsx")
except Exception as e:
    logger.warning(f"Excel file skipped (CSV files were written): {e}")

#-------------------------------------------------------------------------------
# Numbers for the figure legend
#-------------------------------------------------------------------------------
counts = df["age"].value_counts().reindex(AGE_ORDER)
print("\nn nuclei: " + "; ".join(f"{a}, {n:,}" for a, n in counts.items()) + f"; {len(df):,} in total")
print(f"Mice: {df['sample'].nunique()} total; per age group: "
      f"{mice.groupby('age', observed=True).size().to_dict()}")
for metric in RNA_METRICS:
    e = exact[exact["metric"] == metric]
    print(f"{metric}: minima " + ", ".join(f"{x:,.0f}" for x in e["minimum"])
          + " and maxima " + ", ".join(f"{x:,.0f}" for x in e["maximum"]))
print("Lower whisker equals the minimum in every group:",
      bool((exact["whisker_low"] == exact["minimum"]).all()))
print("Adjusted P, Kruskal-Wallis: " + ", ".join(
    f"{r.metric} {r.kruskal_p_BH:.3g}" for r in tests.itertuples()))
