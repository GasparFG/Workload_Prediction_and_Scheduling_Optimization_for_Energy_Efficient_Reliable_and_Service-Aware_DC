"""
compare_solver_vs_baseline.py
==============================
Paired statistical comparison of the MILP solver vs the Round-Robin
baseline across N matched robustness scenarios.

Variables tested:
    1. average server load   (avg_load / avg_util)
    2. total cost             (objective_value / total_cost_$)
    3. total facility energy  (total_facility_energy_kwh / total_energy_kWh)

Tests performed (per variable):
    - Shapiro-Wilk test on the paired differences (normality check)
    - Wilcoxon signed-rank test (primary, non-parametric, paired)
    - Paired t-test (reported alongside, with normality caveat)
    - Effect sizes: matched-pairs rank-biserial correlation (Wilcoxon)
                    and Cohen's d for paired samples (t-test)
    - Sign test (simple robustness check)
    - Relative (%) improvement of solver vs baseline, tested the same way

Multiple comparisons:
    - Holm-Bonferroni correction applied across the 3 variables for both
      the raw and the relative-difference test families.

Input:
    Two CSVs, one row per scenario (case_name), with matching case_name
    columns so rows can be paired. Column names are configurable below.

Usage:
    python compare_solver_vs_baseline.py \
        --solver-csv   solver_summary.csv \
        --baseline-csv rr_summary.csv \
        --key-col      case_name
"""

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Column mapping: (solver_column, baseline_column, display_name, lower_is_better)
# Adjust these names if your CSVs use different headers.
# ---------------------------------------------------------------------------
VARIABLES: List[Tuple[str, str, str, bool]] = [
    ("Utilization",                 "Utilization",
     "Server Utilization",     None),
    ("Cost",          "Cost",         "Total Cost",              True),
    ("Energy", "Energy",      "Total Facility Energy",   True),
    ("switching_cost_$", "switching_cost_$", "switching_cost_$", True),
    ("corrective_maint_$", "corrective_maint_$", "corrective_maint_$", True),
    ("lateness_cost_$", "lateness_cost_$", "lateness_cost_$", True)
]
# lower_is_better: True = lower is better (solver should be lower than baseline)
#                  None = no a-priori direction (just report)


def holm_bonferroni(pvalues: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values
    in the original order."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        adj = max(adj, prev)  # enforce monotonicity
        adj = min(adj, 1.0)
        adjusted[idx] = adj
        prev = adj
    return adjusted.tolist()


def rank_biserial_from_wilcoxon(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation effect size for Wilcoxon.

    r = (sum of ranks for positive differences - sum of ranks for negative
    differences) / total sum of ranks of |diffs| (zeros excluded).
    Range: -1 to 1.
    """
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return float("nan")
    abs_ranks = stats.rankdata(np.abs(diffs))
    pos_sum = abs_ranks[diffs > 0].sum()
    neg_sum = abs_ranks[diffs < 0].sum()
    total = pos_sum + neg_sum
    return (pos_sum - neg_sum) / total


def cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for paired samples = mean(diff) / std(diff, ddof=1)."""
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return float("nan")
    return np.mean(diffs) / sd


def sign_test(diffs: np.ndarray) -> Tuple[int, int, float]:
    """Two-sided exact sign test. Returns (n_positive, n_negative, p_value)."""
    diffs = diffs[diffs != 0]
    n = len(diffs)
    n_pos = int(np.sum(diffs > 0))
    n_neg = n - n_pos
    p = stats.binomtest(min(n_pos, n_neg), n, 0.5,
                        alternative="two-sided").pvalue
    return n_pos, n_neg, p


def run_tests_for_variable(
    solver_vals: np.ndarray,
    baseline_vals: np.ndarray,
    name: str,
    lower_is_better,
) -> Dict:
    """Run the full battery of paired tests on one variable."""
    diffs = solver_vals - baseline_vals  # solver - baseline

    # Shapiro-Wilk on the differences
    if len(diffs) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(diffs)
    else:
        shapiro_stat, shapiro_p = float("nan"), float("nan")

    # Wilcoxon signed-rank (primary)
    try:
        wil_stat, wil_p = stats.wilcoxon(solver_vals, baseline_vals)
    except ValueError:
        wil_stat, wil_p = float("nan"), float("nan")

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(solver_vals, baseline_vals)

    # Effect sizes
    rb_corr = rank_biserial_from_wilcoxon(diffs)
    d = cohens_d_paired(diffs)

    # Sign test
    n_pos, n_neg, sign_p = sign_test(diffs)

    # Relative (%) difference: (solver - baseline) / baseline * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(baseline_vals != 0,
                            (solver_vals - baseline_vals) /
                            np.abs(baseline_vals) * 100.0,
                            np.nan)
    rel_diff = rel_diff[~np.isnan(rel_diff)]

    try:
        wil_rel_stat, wil_rel_p = stats.wilcoxon(
            rel_diff - 0)  # one-sample vs 0
    except ValueError:
        wil_rel_stat, wil_rel_p = float("nan"), float("nan")

    return {
        "name": name,
        "n": len(diffs),
        "mean_solver": np.mean(solver_vals),
        "mean_baseline": np.mean(baseline_vals),
        "mean_diff": np.mean(diffs),
        "mean_rel_diff_pct": np.mean(rel_diff) if len(rel_diff) else float("nan"),
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "wilcoxon_stat": wil_stat,
        "wilcoxon_p": wil_p,
        "ttest_stat": t_stat,
        "ttest_p": t_p,
        "rank_biserial_r": rb_corr,
        "cohens_d": d,
        "sign_n_pos": n_pos,
        "sign_n_neg": n_neg,
        "sign_p": sign_p,
        "wilcoxon_rel_p": wil_rel_p,
        "lower_is_better": lower_is_better,
    }


def interpret(result: Dict, alpha: float = 0.05) -> str:
    """Build a short interpretation line for one variable's results."""
    name = result["name"]
    n = result["n"]
    normal = (not np.isnan(result["shapiro_p"])
              ) and result["shapiro_p"] > alpha
    primary_p = result["wilcoxon_p_adj"]
    sig = primary_p < alpha if not np.isnan(primary_p) else False

    direction = "lower" if result["mean_diff"] < 0 else "higher"
    lib = result["lower_is_better"]

    lines = []
    lines.append(
        f"{name} (n={n}): solver mean={result['mean_solver']:.4g}, "
        f"baseline mean={result['mean_baseline']:.4g}, "
        f"mean relative change={result['mean_rel_diff_pct']:.2f}%"
    )
    lines.append(
        f"  Shapiro-Wilk on differences: p={result['shapiro_p']:.4f} "
        f"({'looks normal' if normal else 'non-normal — favor Wilcoxon'})"
    )
    lines.append(
        f"  Wilcoxon signed-rank: p={result['wilcoxon_p']:.4f} "
        f"(Holm-adjusted p={primary_p:.4f}), "
        f"rank-biserial r={result['rank_biserial_r']:.3f}"
    )
    lines.append(
        f"  Paired t-test:        p={result['ttest_p']:.4f} "
        f"(Holm-adjusted p={result['ttest_p_adj']:.4f}), "
        f"Cohen's d={result['cohens_d']:.3f}"
    )
    lines.append(
        f"  Sign test: {result['sign_n_pos']} pairs favor solver-higher, "
        f"{result['sign_n_neg']} favor baseline-higher, p={result['sign_p']:.4f}"
    )

    if sig:
        verdict = f"Solver is significantly {direction} than baseline"
        if lib is True:
            verdict += " (this is the desired direction — solver wins)" if direction == "lower" \
                else " (this is the UNDESIRED direction — baseline wins)"
        lines.append(f"  -> {verdict} (alpha={alpha}).")
    else:
        lines.append(
            f"  -> No statistically significant difference detected (alpha={alpha}).")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Paired stats: MILP solver vs round-robin baseline")
    parser.add_argument("--solver-csv", required=True,
                        help="CSV with one row per scenario for the MILP solver results.")
    parser.add_argument("--baseline-csv", required=True,
                        help="CSV with one row per scenario for the round-robin baseline results.")
    parser.add_argument("--key-col", default="case_name",
                        help="Column used to match/pair rows between the two CSVs (default: case_name).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05).")
    parser.add_argument("--out-csv", default=None,
                        help="Optional path to write a results table CSV.")
    args = parser.parse_args()

    solver_df = pd.read_csv(args.solver_csv)
    baseline_df = pd.read_csv(args.baseline_csv)

    if args.key_col not in solver_df.columns or args.key_col not in baseline_df.columns:
        raise SystemExit(
            f"Key column '{args.key_col}' not found in one of the input CSVs. "
            f"Solver columns: {list(solver_df.columns)}. "
            f"Baseline columns: {list(baseline_df.columns)}."
        )

    merged = pd.merge(
        solver_df, baseline_df, on=args.key_col,
        suffixes=("_solver", "_baseline"), how="inner"
    )
    print(f"Matched {len(merged)} scenario pairs on '{args.key_col}'.\n")

    results = []
    for solver_col, baseline_col, name, lib in VARIABLES:
        sc = solver_col if solver_col in merged.columns else f"{solver_col}_solver"
        bc = baseline_col if baseline_col in merged.columns else f"{baseline_col}_baseline"

        if sc not in merged.columns or bc not in merged.columns:
            print(f"WARNING: could not find columns for '{name}' "
                  f"(looked for '{sc}' and '{bc}'). Skipping.\n")
            continue

        sub = merged[[sc, bc]].dropna()
        if len(sub) < 2:
            print(
                f"WARNING: not enough non-missing pairs for '{name}'. Skipping.\n")
            continue

        res = run_tests_for_variable(
            sub[sc].to_numpy(dtype=float),
            sub[bc].to_numpy(dtype=float),
            name, lib,
        )
        results.append(res)

    if not results:
        raise SystemExit("No variables could be tested. Check column names.")

    # Holm-Bonferroni correction across the 3 (or fewer) variables
    wil_ps = [r["wilcoxon_p"] for r in results]
    t_ps = [r["ttest_p"] for r in results]
    wil_adj = holm_bonferroni(wil_ps)
    t_adj = holm_bonferroni(t_ps)
    for r, wa, ta in zip(results, wil_adj, t_adj):
        r["wilcoxon_p_adj"] = wa
        r["ttest_p_adj"] = ta

    print("=" * 70)
    print("PAIRED COMPARISON: MILP SOLVER vs ROUND-ROBIN BASELINE")
    print("(Holm-Bonferroni correction applied across the tested variables)")
    print("=" * 70)
    for r in results:
        print()
        print(interpret(r, alpha=args.alpha))

    if args.out_csv:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.out_csv, index=False)
        print(f"\nFull results table written to: {args.out_csv}")


if __name__ == "__main__":
    main()
