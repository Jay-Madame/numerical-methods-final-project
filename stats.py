"""
stats.py — Statistical Analysis of Gaussian Elimination Benchmark Results
Group: Jewell Callahan, Carson Milano, Tyler Evans, Damian McCone

Reads results.csv produced by main.py and performs:
  1. Descriptive statistics per algorithm per matrix size
  2. One-way ANOVA across algorithms (per matrix size)
  3. Tukey HSD post-hoc test to identify which pairs differ significantly
  4. Writes a full summary to stats_report.txt

Usage:
    python stats.py              # reads results.csv by default
    python stats.py my_data.csv  # custom CSV path
"""

import sys
import csv
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd
from itertools import combinations


CSV_FILE    = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
REPORT_FILE = "stats_report.txt"
ALPHA       = 0.05   # significance level


# ── Load Data ─────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "OK"].copy()
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    df = df.dropna(subset=["time_ms"])
    return df


# ── Descriptive Stats ─────────────────────────────────────────────────────────

def descriptive(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["matrix_size", "algorithm"])["time_ms"]
        .agg(n="count", mean="mean", std="std", median="median",
             min="min", max="max")
        .round(4)
        .reset_index()
    )


# ── One-Way ANOVA + Tukey HSD (per matrix size) ───────────────────────────────

def anova_per_size(df: pd.DataFrame):
    sizes   = sorted(df["matrix_size"].unique())
    results = []

    for n in sizes:
        sub   = df[df["matrix_size"] == n]
        algos = sorted(sub["algorithm"].unique())
        groups = [sub[sub["algorithm"] == a]["time_ms"].values for a in algos]

        # One-way ANOVA
        f_stat, p_val = f_oneway(*groups)
        significant   = p_val < ALPHA

        # Tukey HSD post-hoc (only meaningful if ANOVA is significant)
        tukey_rows = []
        if significant and len(groups) >= 2:
            tukey = tukey_hsd(*groups)
            for i, j in combinations(range(len(algos)), 2):
                p = tukey.pvalue[i][j]
                tukey_rows.append({
                    "matrix_size": n,
                    "group_A":     algos[i],
                    "group_B":     algos[j],
                    "p_value":     round(p, 6),
                    "significant": p < ALPHA,
                })

        results.append({
            "matrix_size": n,
            "algorithms":  algos,
            "groups":      groups,
            "f_stat":      round(f_stat, 4),
            "p_value":     round(p_val, 6),
            "significant": significant,
            "tukey":       tukey_rows,
        })

    return results


# ── Report Writer ─────────────────────────────────────────────────────────────

def write_report(desc: pd.DataFrame, anova_results: list, path: str):
    lines = []
    w     = 80

    def rule(c="="):  lines.append(c * w)
    def blank():      lines.append("")
    def h(t):
        rule()
        lines.append(f"  {t}")
        rule()

    h("GAUSSIAN ELIMINATION — STATISTICAL ANALYSIS REPORT")
    lines.append(f"  Significance level: α = {ALPHA}")
    lines.append(f"  Test: One-way ANOVA + Tukey HSD post-hoc")
    blank()

    # ── Descriptive Stats ──────────────────────────────────────────────────
    rule()
    lines.append("  DESCRIPTIVE STATISTICS  (times in ms)")
    rule()

    sizes = sorted(desc["matrix_size"].unique())
    for n in sizes:
        sub = desc[desc["matrix_size"] == n].reset_index(drop=True)
        lines.append(f"\n  Matrix size: {n}×{n}")
        lines.append(f"  {'Algorithm':<32} {'n':>4} {'Mean':>10} {'Std':>10} "
                     f"{'Median':>10} {'Min':>10} {'Max':>10}")
        lines.append("  " + "-" * 78)
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['algorithm']:<32} {int(row['n']):>4} "
                f"{row['mean']:>10.4f} {row['std']:>10.4f} "
                f"{row['median']:>10.4f} {row['min']:>10.4f} {row['max']:>10.4f}"
            )

    blank()

    # ── ANOVA Results ──────────────────────────────────────────────────────
    rule()
    lines.append("  ONE-WAY ANOVA RESULTS")
    rule()
    lines.append(f"\n  H₀: All algorithms have equal mean runtime for a given matrix size.")
    lines.append(f"  H₁: At least one algorithm differs significantly.\n")
    lines.append(f"  {'Size':>6}  {'F-statistic':>14}  {'p-value':>12}  {'Significant?':>14}  Conclusion")
    lines.append("  " + "-" * 74)

    for r in anova_results:
        sig  = "YES ✓" if r["significant"] else "NO"
        conc = "Reject H₀" if r["significant"] else "Fail to reject H₀"
        lines.append(
            f"  {r['matrix_size']:>6}  {r['f_stat']:>14.4f}  "
            f"{r['p_value']:>12.6f}  {sig:>14}  {conc}"
        )

    blank()

    # ── Tukey HSD ─────────────────────────────────────────────────────────
    rule()
    lines.append("  TUKEY HSD POST-HOC TEST  (pairwise comparisons)")
    rule()
    lines.append(f"\n  Only shown for matrix sizes where ANOVA was significant (p < {ALPHA}).\n")

    any_tukey = False
    for r in anova_results:
        if not r["tukey"]:
            continue
        any_tukey = True
        lines.append(f"  Matrix size: {r['matrix_size']}×{r['matrix_size']}")
        lines.append(f"  {'Group A':<32} {'Group B':<32} {'p-value':>10}  {'Significant?':>14}")
        lines.append("  " + "-" * 90)
        for t in r["tukey"]:
            sig = "YES ✓" if t["significant"] else "NO"
            lines.append(
                f"  {t['group_A']:<32} {t['group_B']:<32} "
                f"{t['p_value']:>10.6f}  {sig:>14}"
            )
        blank()

    if not any_tukey:
        lines.append("  No significant ANOVA results — Tukey HSD not applicable.")
        blank()

    # ── Overall Conclusion ─────────────────────────────────────────────────
    rule()
    lines.append("  OVERALL CONCLUSION")
    rule()
    sig_sizes = [r["matrix_size"] for r in anova_results if r["significant"]]
    if sig_sizes:
        lines.append(f"\n  Statistically significant differences in runtime were found at")
        lines.append(f"  matrix sizes: {sig_sizes}")
        lines.append(f"\n  This means the choice of algorithm has a measurable impact on")
        lines.append(f"  performance at these sizes. Refer to Tukey HSD above for which")
        lines.append(f"  specific pairs differ.")
    else:
        lines.append(f"\n  No statistically significant differences were found at α={ALPHA}.")
        lines.append(f"  All algorithms perform comparably across tested matrix sizes.")
    blank()
    rule()

    report = "\n".join(lines)
    with open(path, "w") as f:
        f.write(report)
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  Loading '{CSV_FILE}'...")
    df = load(CSV_FILE)

    print(f"  {len(df)} valid observations across "
          f"{df['algorithm'].nunique()} algorithms, "
          f"{df['matrix_size'].nunique()} matrix sizes.\n")

    desc          = descriptive(df)
    anova_results = anova_per_size(df)
    report        = write_report(desc, anova_results, REPORT_FILE)

    print(report)
    print(f"\n  Full report saved → '{REPORT_FILE}'\n")
