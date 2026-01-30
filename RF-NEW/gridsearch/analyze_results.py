#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_results.py

Publication-friendly post-processing for Random Forest experiments.

Modes:
1) Single-root mode:
   - Summarize PRE vs POST results within one results_root
   - Summarize best configuration per balancing strategy (compact pivot + optional details)
   - Generate a clean Markdown report (GitHub-friendly)
   - Save consolidated CSVs and plots

2) Compare-two-roots mode:
   - Compare "with label features" vs "no label features"
   - Provide deltas:
       a) POST - PRE within each scenario
       b) WITH_LABELS - NO_LABELS for PRE and POST
   - Save consolidated CSVs + plots + Markdown report

Expected folder structure for each root:
results_root/
  pre/
    results.json
    cv_best_per_strategy.csv
    cv_results.csv
    REPORT.txt
  post/
    results.json
    cv_best_per_strategy.csv
    cv_results.csv
    REPORT.txt


    # Modo 1: Analisar um results_root (PRE vs POST)
python3 analyze_results.py outputs/rf__labels-full__seed-42

# Modo 2: Comparar dois results_root (full vs count_only)
python3 analyze_results.py outputs/rf__labels-full__seed-42 outputs/rf__labels-count_only__seed-42 \
  --scenario-with "with_labels" --scenario-without "no_labels" --out-dir analysis_labels_impact
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class ExperimentResult:
    feature_set: str
    n_features: int
    search_method: str
    elapsed_time_minutes: float
    best_cv_f1_macro: float
    best_params: Dict[str, Any]
    test_f1_macro: float
    test_balanced_accuracy: float
    artifacts_dir: str


# -----------------------------
# IO helpers
# -----------------------------
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_result(exp_dir: str, feature_set: str) -> ExperimentResult:
    results_path = os.path.join(exp_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing results.json: {results_path}")

    data = load_json(results_path)

    # Robust metric loading:
    # 1) Prefer nested "test_metrics" (current pipeline)
    # 2) Fallback to top-level keys (legacy outputs)
    tm = data.get("test_metrics", {}) if isinstance(data.get("test_metrics", {}), dict) else {}

    test_f1_macro = tm.get("test_f1_macro", data.get("test_f1_macro", None))
    test_bal_acc = tm.get("test_balanced_accuracy", data.get("test_balanced_accuracy", None))

    if test_f1_macro is None or test_bal_acc is None:
        available_keys = sorted(list(data.keys()))
        available_tm_keys = sorted(list(tm.keys())) if isinstance(tm, dict) else []
        raise ValueError(
            "Missing test metrics in results.json. "
            f"Top-level keys={available_keys} | test_metrics keys={available_tm_keys}"
        )

    return ExperimentResult(
        feature_set=feature_set,
        n_features=int(data.get("n_features", 0)),
        search_method=str(data.get("search_method", "")),
        elapsed_time_minutes=float(data.get("elapsed_time_minutes", 0.0)),
        best_cv_f1_macro=float(data.get("best_cv_f1_macro", 0.0)),
        best_params=dict(data.get("best_params", {})),
        test_f1_macro=float(test_f1_macro),
        test_balanced_accuracy=float(test_bal_acc),
        artifacts_dir=str(data.get("artifacts_dir", exp_dir)),
    )

def load_best_per_strategy(exp_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(exp_dir, "cv_best_per_strategy.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def extract_strategy(best_params: Dict[str, Any]) -> str:
    return str(best_params.get("clf__strategy", "unknown"))


# -----------------------------
# Table builders (single root)
# -----------------------------
def build_consolidated_tables(results_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pre_dir = os.path.join(results_root, "pre")
    post_dir = os.path.join(results_root, "post")

    pre_res = load_experiment_result(pre_dir, "pre")
    post_res = load_experiment_result(post_dir, "post")

    summary_rows: List[Dict[str, Any]] = []
    for r in [pre_res, post_res]:
        summary_rows.append(
            {
                "feature_set": r.feature_set,
                "n_features": r.n_features,
                "search_method": r.search_method,
                "elapsed_time_minutes": r.elapsed_time_minutes,
                "best_cv_f1_macro": r.best_cv_f1_macro,
                "best_strategy": extract_strategy(r.best_params),
                "test_f1_macro": r.test_f1_macro,
                "test_balanced_accuracy": r.test_balanced_accuracy,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["test_f1_macro", "test_balanced_accuracy"], ascending=False
    )

    pre_strat = load_best_per_strategy(pre_dir)
    post_strat = load_best_per_strategy(post_dir)

    strat_rows: List[pd.DataFrame] = []
    for feature_set, df in [("pre", pre_strat), ("post", post_strat)]:
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, "feature_set", feature_set)
        strat_rows.append(tmp)

    strat_df = pd.concat(strat_rows, ignore_index=True) if strat_rows else pd.DataFrame()
    return summary_df, strat_df


# -----------------------------
# Table builders (two roots)
# -----------------------------
def build_summary_table_for_root(results_root: str, scenario: str) -> pd.DataFrame:
    summary_df, _ = build_consolidated_tables(results_root)
    out = summary_df.copy()
    out.insert(0, "scenario", scenario)
    return out


def build_strategy_table_for_root(results_root: str, scenario: str) -> pd.DataFrame:
    _, strat_df = build_consolidated_tables(results_root)
    out = strat_df.copy()
    if out.empty:
        return out
    out.insert(0, "scenario", scenario)
    return out


# -----------------------------
# Formatting helpers
# -----------------------------
def format_float_cols(df: pd.DataFrame, cols: List[str], decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(decimals)
    return out


def build_strategy_pivot(strat_df: pd.DataFrame, index_cols: List[str]) -> pd.DataFrame:
    """
    Build a compact table:
      rows = index_cols (e.g., [scenario, strategy])
      columns = feature_set (pre/post)
      values = best mean_test_score (macro-F1)
    """
    required = {"feature_set", "param_clf__strategy", "mean_test_score"}
    if strat_df.empty or not required.issubset(set(strat_df.columns)):
        return pd.DataFrame()

    tmp = strat_df.copy()
    tmp["mean_test_score"] = pd.to_numeric(tmp["mean_test_score"], errors="coerce")

    pivot = (
        tmp.pivot_table(
            index=index_cols + ["param_clf__strategy"],
            columns="feature_set",
            values="mean_test_score",
            aggfunc="max",
        )
        .reset_index()
        .rename(columns={"param_clf__strategy": "strategy"})
    )

    sort_cols = [c for c in ["post", "pre"] if c in pivot.columns]
    if sort_cols:
        pivot = pivot.sort_values(sort_cols, ascending=False)

    return pivot


def compute_pre_post_delta(summary_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Compute deltas (POST - PRE) for key metrics within a summary_df that has feature_set {pre, post}.
    """
    def _get(set_name: str, col: str) -> Optional[float]:
        row = summary_df[summary_df["feature_set"] == set_name]
        if row.empty or col not in row.columns:
            return None
        return float(row.iloc[0][col])

    pre_f1 = _get("pre", "test_f1_macro")
    post_f1 = _get("post", "test_f1_macro")
    pre_bal = _get("pre", "test_balanced_accuracy")
    post_bal = _get("post", "test_balanced_accuracy")

    delta_f1 = (post_f1 - pre_f1) if (pre_f1 is not None and post_f1 is not None) else None
    delta_bal = (post_bal - pre_bal) if (pre_bal is not None and post_bal is not None) else None

    return {"delta_test_f1_macro": delta_f1, "delta_test_bal_acc": delta_bal}


def compute_label_feature_impact(
    combined_summary: pd.DataFrame,
    scenario_with: str,
    scenario_without: str,
) -> pd.DataFrame:
    """
    Compute WITH_LABELS - NO_LABELS impact for each feature_set (pre/post) on test metrics.
    Returns a compact table for the report.
    """
    needed = {"scenario", "feature_set", "test_f1_macro", "test_balanced_accuracy"}
    if not needed.issubset(set(combined_summary.columns)):
        return pd.DataFrame()

    df = combined_summary.copy()
    df = df[df["scenario"].isin([scenario_with, scenario_without])].copy()

    pivot_f1 = df.pivot_table(
        index="feature_set",
        columns="scenario",
        values="test_f1_macro",
        aggfunc="first",
    )
    pivot_bal = df.pivot_table(
        index="feature_set",
        columns="scenario",
        values="test_balanced_accuracy",
        aggfunc="first",
    )

    rows: List[Dict[str, Any]] = []
    for fs in sorted(df["feature_set"].unique()):
        f1_with = pivot_f1.get(scenario_with, pd.Series(dtype=float)).get(fs)
        f1_wo = pivot_f1.get(scenario_without, pd.Series(dtype=float)).get(fs)
        bal_with = pivot_bal.get(scenario_with, pd.Series(dtype=float)).get(fs)
        bal_wo = pivot_bal.get(scenario_without, pd.Series(dtype=float)).get(fs)

        row = {"set": fs}
        if pd.notna(f1_with) and pd.notna(f1_wo):
            row["delta_test_f1_macro"] = float(f1_with - f1_wo)
        else:
            row["delta_test_f1_macro"] = None

        if pd.notna(bal_with) and pd.notna(bal_wo):
            row["delta_test_bal_acc"] = float(bal_with - bal_wo)
        else:
            row["delta_test_bal_acc"] = None

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = format_float_cols(out, ["delta_test_f1_macro", "delta_test_bal_acc"], decimals=4)
    return out


# -----------------------------
# Plotting
# -----------------------------
def _add_bar_labels(ax: plt.Axes, fmt: str = "{:.4f}") -> None:
    for p in ax.patches:
        height = p.get_height()
        if height is None:
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_pre_post_summary(summary_df: pd.DataFrame, out_dir: str, filename: str) -> str:
    df = format_float_cols(summary_df, ["test_f1_macro"], decimals=4)

    plt.figure(figsize=(6.5, 4))
    ax = sns.barplot(data=df, x="feature_set", y="test_f1_macro", palette="viridis")
    ax.set_title("Test Macro-F1: PRE vs POST")
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("")
    ax.set_ylim(0, max(0.01, float(df["test_f1_macro"].max()) + 0.08))
    _add_bar_labels(ax, fmt="{:.4f}")
    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def plot_strategy_cv_single(strat_df: pd.DataFrame, out_dir: str, filename: str) -> Optional[str]:
    if strat_df.empty:
        return None

    required = {"param_clf__strategy", "mean_test_score", "feature_set"}
    if not required.issubset(set(strat_df.columns)):
        return None

    df = strat_df.copy()
    df["mean_test_score"] = pd.to_numeric(df["mean_test_score"], errors="coerce")

    pivot = build_strategy_pivot(df, index_cols=[])
    order = pivot["strategy"].tolist() if (not pivot.empty and "strategy" in pivot.columns) else None

    plt.figure(figsize=(9.5, 4))
    ax = sns.barplot(
        data=df,
        x="param_clf__strategy",
        y="mean_test_score",
        hue="feature_set",
        order=order,
        palette={"pre": "#2a9d8f", "post": "#e76f51"},
    )
    ax.set_title("Best CV Macro-F1 per Balancing Strategy")
    ax.set_ylabel("CV macro-F1 (mean_test_score)")
    ax.set_xlabel("Strategy")
    ax.set_ylim(0, max(0.01, float(df["mean_test_score"].max()) + 0.08))
    _add_bar_labels(ax, fmt="{:.4f}")

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


def plot_compare_metric(
    combined_summary: pd.DataFrame,
    out_dir: str,
    metric_col: str,
    title: str,
    filename: str,
) -> str:
    df = combined_summary.copy()
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col])

    plt.figure(figsize=(8.8, 4.2))
    ax = sns.barplot(
        data=df,
        x="feature_set",
        y=metric_col,
        hue="scenario",
        palette="Set2",
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(metric_col)
    ax.set_ylim(0, max(0.01, float(df[metric_col].max()) + 0.08))
    _add_bar_labels(ax, fmt="{:.4f}")
    plt.tight_layout()

    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=220)
    plt.close()
    return out_path


# -----------------------------
# Report writers
# -----------------------------
def write_markdown_report_single_root(
    results_root: str,
    summary_df: pd.DataFrame,
    strat_df: pd.DataFrame,
    plot1: str,
    plot2: Optional[str],
) -> str:
    out_path = os.path.join(results_root, "FINAL_REPORT.md")
    deltas = compute_pre_post_delta(summary_df)

    summary_cols = [
        "feature_set",
        "n_features",
        "search_method",
        "elapsed_time_minutes",
        "best_cv_f1_macro",
        "best_strategy",
        "test_f1_macro",
        "test_balanced_accuracy",
    ]
    summary_cols = [c for c in summary_cols if c in summary_df.columns]
    summary_view = summary_df[summary_cols].copy().rename(
        columns={
            "feature_set": "set",
            "elapsed_time_minutes": "time_min",
            "best_cv_f1_macro": "cv_f1_macro",
            "best_strategy": "best_balance",
            "test_balanced_accuracy": "test_bal_acc",
        }
    )
    summary_view = format_float_cols(summary_view, ["time_min", "cv_f1_macro", "test_f1_macro", "test_bal_acc"], 4)

    strat_pivot = build_strategy_pivot(strat_df, index_cols=[])
    if not strat_pivot.empty:
        strat_pivot = format_float_cols(strat_pivot, [c for c in ["pre", "post"] if c in strat_pivot.columns], 4)

    detail_keep = [
        "feature_set",
        "param_clf__strategy",
        "mean_test_score",
        "std_test_score",
        "param_clf__n_estimators",
        "param_clf__max_depth",
        "param_clf__max_features",
        "param_clf__min_samples_split",
        "param_clf__min_samples_leaf",
    ]
    detail_keep = [c for c in detail_keep if c in strat_df.columns]
    strat_detail = strat_df[detail_keep].copy() if (not strat_df.empty and detail_keep) else pd.DataFrame()
    if not strat_detail.empty:
        strat_detail = strat_detail.rename(
            columns={
                "feature_set": "set",
                "param_clf__strategy": "strategy",
                "mean_test_score": "cv_f1_macro",
                "std_test_score": "cv_f1_std",
            }
        )
        strat_detail = format_float_cols(strat_detail, ["cv_f1_macro", "cv_f1_std"], 4)

    lines: List[str] = []
    lines.append("# Random Forest Results Report\n")
    lines.append("**Task:** Predict `PR_lifetime` (multiclass: 1–5)\n")
    lines.append("**Experiments:** PRE vs POST feature availability\n")
    lines.append("**Balancing strategies:** none / undersample / oversample\n")
    lines.append("\n---\n")

    lines.append("## Key takeaways\n")
    if deltas["delta_test_f1_macro"] is not None:
        lines.append(f"- **Test Macro-F1 (POST − PRE):** {deltas['delta_test_f1_macro']:+.4f}\n")
    if deltas["delta_test_bal_acc"] is not None:
        lines.append(f"- **Test Balanced Accuracy (POST − PRE):** {deltas['delta_test_bal_acc']:+.4f}\n")
    lines.append("- **CV selection metric:** macro-F1 (more robust under class imbalance)\n")
    lines.append("\n---\n")

    lines.append("## Main results (Test set)\n")
    lines.append(summary_view.to_markdown(index=False, tablefmt="github"))
    lines.append("\n")
    lines.append(f"![Test Macro-F1 PRE vs POST]({os.path.basename(plot1)})\n")

    if not strat_pivot.empty:
        lines.append("\n---\n")
        lines.append("## Best CV Macro-F1 by balancing strategy\n")
        lines.append(strat_pivot.to_markdown(index=False, tablefmt="github"))
        lines.append("\n")
        if plot2 is not None:
            lines.append(f"![Best CV per Strategy]({os.path.basename(plot2)})\n")

    if not strat_detail.empty:
        lines.append("\n---\n")
        lines.append("## Detailed hyperparameters (best per strategy)\n")
        lines.append("<details>\n<summary>Click to expand</summary>\n\n")
        lines.append(strat_detail.to_markdown(index=False, tablefmt="github"))
        lines.append("\n</details>\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


def write_markdown_report_compare(
    out_dir: str,
    combined_summary: pd.DataFrame,
    combined_strat: pd.DataFrame,
    scenario_with: str,
    scenario_without: str,
    plot_f1: str,
    plot_bal: str,
) -> str:
    out_path = os.path.join(out_dir, "FINAL_REPORT_COMPARE.md")

    # Compact main table
    keep = [
        "scenario",
        "feature_set",
        "n_features",
        "search_method",
        "elapsed_time_minutes",
        "best_cv_f1_macro",
        "best_strategy",
        "test_f1_macro",
        "test_balanced_accuracy",
    ]
    keep = [c for c in keep if c in combined_summary.columns]
    view = combined_summary[keep].copy().rename(
        columns={
            "feature_set": "set",
            "elapsed_time_minutes": "time_min",
            "best_cv_f1_macro": "cv_f1_macro",
            "test_balanced_accuracy": "test_bal_acc",
            "best_strategy": "best_balance",
        }
    )
    view = format_float_cols(view, ["time_min", "cv_f1_macro", "test_f1_macro", "test_bal_acc"], 4)

    # POST - PRE deltas per scenario
    deltas_rows: List[Dict[str, Any]] = []
    for scenario in sorted(view["scenario"].unique()):
        sub = combined_summary[combined_summary["scenario"] == scenario].copy()
        d = compute_pre_post_delta(sub)
        deltas_rows.append(
            {
                "scenario": scenario,
                "delta_post_minus_pre_test_f1_macro": d["delta_test_f1_macro"],
                "delta_post_minus_pre_test_bal_acc": d["delta_test_bal_acc"],
            }
        )
    deltas_df = pd.DataFrame(deltas_rows)
    deltas_df = format_float_cols(
        deltas_df,
        ["delta_post_minus_pre_test_f1_macro", "delta_post_minus_pre_test_bal_acc"],
        4,
    )

    # WITH - WITHOUT label features per set
    impact_df = compute_label_feature_impact(
        combined_summary=combined_summary,
        scenario_with=scenario_with,
        scenario_without=scenario_without,
    )

    # Strategy pivot with scenario as part of the index
    strat_pivot = build_strategy_pivot(combined_strat, index_cols=["scenario"]) if not combined_strat.empty else pd.DataFrame()
    if not strat_pivot.empty:
        strat_pivot = format_float_cols(strat_pivot, [c for c in ["pre", "post"] if c in strat_pivot.columns], 4)

    lines: List[str] = []
    lines.append("# Random Forest Comparison Report\n")
    lines.append("This report compares **two experiment roots** to estimate the impact of using `CONT_label_*` features.\n")
    lines.append(f"- **WITH label features scenario:** `{scenario_with}`\n")
    lines.append(f"- **WITHOUT label features scenario:** `{scenario_without}`\n")
    lines.append("\n---\n")

    lines.append("## Main results (Test set)\n")
    lines.append(view.to_markdown(index=False, tablefmt="github"))
    lines.append("\n---\n")

    lines.append("## PRE vs POST deltas (within each scenario)\n")
    lines.append(deltas_df.to_markdown(index=False, tablefmt="github"))
    lines.append("\n---\n")

    lines.append("## Impact of label features (WITH − WITHOUT)\n")
    lines.append("Positive values indicate that adding `CONT_label_*` **improved** the metric.\n\n")
    if not impact_df.empty:
        lines.append(impact_df.to_markdown(index=False, tablefmt="github"))
    else:
        lines.append("_Impact table could not be computed (missing columns or missing scenarios)._")
    lines.append("\n\n---\n")

    lines.append("## Plots\n")
    lines.append(f"![Test Macro-F1 comparison]({os.path.basename(plot_f1)})\n")
    lines.append(f"![Test Balanced Accuracy comparison]({os.path.basename(plot_bal)})\n")

    if not strat_pivot.empty:
        lines.append("\n---\n")
        lines.append("## Best CV Macro-F1 by balancing strategy (per scenario)\n")
        lines.append(strat_pivot.to_markdown(index=False, tablefmt="github"))
        lines.append("\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


# -----------------------------
# Entry points
# -----------------------------
def run_single_root(results_root: str) -> None:
    summary_df, strat_df = build_consolidated_tables(results_root)

    summary_csv = os.path.join(results_root, "final_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    strat_csv = os.path.join(results_root, "final_best_per_strategy.csv")
    strat_df.to_csv(strat_csv, index=False)

    plot1 = plot_pre_post_summary(summary_df, results_root, filename="plot_test_f1_pre_vs_post.png")
    plot2 = plot_strategy_cv_single(strat_df, results_root, filename="plot_cv_best_per_strategy.png")

    report_path = write_markdown_report_single_root(results_root, summary_df, strat_df, plot1, plot2)

    print("✅ Final artifacts created (single root):")
    print(f"- {summary_csv}")
    print(f"- {strat_csv}")
    print(f"- {plot1}")
    if plot2:
        print(f"- {plot2}")
    print(f"- {report_path}")


def run_compare_roots(
    root_with: str,
    root_without: str,
    scenario_with: str,
    scenario_without: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    s_with = build_summary_table_for_root(root_with, scenario_with)
    s_wo = build_summary_table_for_root(root_without, scenario_without)
    combined_summary = pd.concat([s_with, s_wo], ignore_index=True)

    t_with = build_strategy_table_for_root(root_with, scenario_with)
    t_wo = build_strategy_table_for_root(root_without, scenario_without)
    combined_strat = pd.concat([t_with, t_wo], ignore_index=True) if (not t_with.empty or not t_wo.empty) else pd.DataFrame()

    # Save CSVs
    combined_summary_csv = os.path.join(out_dir, "final_summary_compare.csv")
    combined_summary.to_csv(combined_summary_csv, index=False)

    combined_strat_csv = os.path.join(out_dir, "final_best_per_strategy_compare.csv")
    combined_strat.to_csv(combined_strat_csv, index=False)

    # Plots
    plot_f1 = plot_compare_metric(
        combined_summary=combined_summary,
        out_dir=out_dir,
        metric_col="test_f1_macro",
        title="Test Macro-F1: WITH vs WITHOUT label features",
        filename="plot_compare_test_f1_macro.png",
    )
    plot_bal = plot_compare_metric(
        combined_summary=combined_summary,
        out_dir=out_dir,
        metric_col="test_balanced_accuracy",
        title="Test Balanced Accuracy: WITH vs WITHOUT label features",
        filename="plot_compare_test_balanced_accuracy.png",
    )

    report_path = write_markdown_report_compare(
        out_dir=out_dir,
        combined_summary=combined_summary,
        combined_strat=combined_strat,
        scenario_with=scenario_with,
        scenario_without=scenario_without,
        plot_f1=plot_f1,
        plot_bal=plot_bal,
    )

    print("✅ Final artifacts created (comparison):")
    print(f"- {combined_summary_csv}")
    print(f"- {combined_strat_csv}")
    print(f"- {plot_f1}")
    print(f"- {plot_bal}")
    print(f"- {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Random Forest experiment outputs (single root or compare two roots)."
    )
    parser.add_argument(
        "results_root",
        nargs="+",
        help="One results_root (single mode) or two results_roots (compare mode).",
    )
    parser.add_argument(
        "--scenario-with",
        default="with_label_features",
        help="Scenario name for the root that includes CONT_label_* features (compare mode).",
    )
    parser.add_argument(
        "--scenario-without",
        default="no_label_features",
        help="Scenario name for the root that excludes CONT_label_* features (compare mode).",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis_compare",
        help="Output directory for comparison artifacts (compare mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if len(args.results_root) == 1:
        run_single_root(args.results_root[0])
        return

    if len(args.results_root) != 2:
        raise SystemExit("Usage: python analyze_results.py <results_root> OR <root_with> <root_without>")

    root_with, root_without = args.results_root
    run_compare_roots(
        root_with=root_with,
        root_without=root_without,
        scenario_with=args.scenario_with,
        scenario_without=args.scenario_without,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
