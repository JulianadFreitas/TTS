#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reporting and artifact export utilities."""

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline


def best_per_strategy_from_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Return the best CV row per balancing strategy."""
    if "param_clf__strategy" not in cv_results.columns:
        return pd.DataFrame()

    keep_cols = [
        "param_clf__strategy",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "param_clf__n_estimators",
        "param_clf__max_depth",
        "param_clf__max_features",
        "param_clf__min_samples_split",
        "param_clf__min_samples_leaf",
    ]
    keep_cols = [c for c in keep_cols if c in cv_results.columns]
    compact = cv_results[keep_cols].copy()

    best_rows = (
        compact.sort_values(["param_clf__strategy", "mean_test_score"], ascending=[True, False])
        .groupby("param_clf__strategy", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best_rows


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def export_experiment_artifacts(
    out_dir: str,
    mode: str,
    label_features_mode: str,
    best_params: Dict[str, Any],
    best_cv_score: float,
    test_metrics: Dict[str, Any],
    features_used: List[str],
    cv_results: pd.DataFrame,
    best_per_strategy: pd.DataFrame,
    model: Pipeline,
    elapsed_time: float,
) -> None:
    """Export all experiment results and artifacts."""
    ensure_dir(out_dir)

    # Save features and CV artifacts
    pd.Series(features_used, name="feature").to_csv(
        os.path.join(out_dir, "features_used.csv"), index=False
    )
    cv_results.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)
    best_per_strategy.to_csv(os.path.join(out_dir, "cv_best_per_strategy.csv"), index=False)
    dump(model, os.path.join(out_dir, "best_model.joblib"))

    payload = {
        "feature_set": mode,
        "label_features_mode": label_features_mode,
        "n_features": int(len(features_used)),
        "search_method": "full (GridSearchCV)",
        "elapsed_time_minutes": round(elapsed_time / 60, 2),
        "best_cv_f1_macro": float(best_cv_score),
        "best_params": best_params,
        "test_metrics": {
            "test_f1_macro": float(test_metrics["test_f1_macro"]),
            "test_balanced_accuracy": float(test_metrics["test_balanced_accuracy"]),
        },
        "artifacts_dir": out_dir,
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Human-readable report
    report_lines = []
    report_lines.append(f"Feature set: {mode}")
    report_lines.append(f"Label features mode: {label_features_mode}")
    report_lines.append(f"Search method: full (GridSearchCV)")
    report_lines.append(f"Number of features: {len(features_used)}")
    report_lines.append(f"Elapsed time: {elapsed_time/60:.2f} minutes")
    report_lines.append(f"Best CV macro-F1: {best_cv_score:.6f}")
    report_lines.append(f"Best params: {best_params}")
    report_lines.append("")
    report_lines.append("Test metrics:")
    report_lines.append(f"- test_f1_macro: {test_metrics['test_f1_macro']:.6f}")
    report_lines.append(f"- test_balanced_accuracy: {test_metrics['test_balanced_accuracy']:.6f}")
    report_lines.append("")
    report_lines.append("Best per strategy (CV):")
    report_lines.append(best_per_strategy.to_string(index=False))
    report_lines.append("")
    report_lines.append("Classification report:")
    report_lines.append(test_metrics["classification_report"])
    report_lines.append("")
    report_lines.append("Confusion matrix:")
    report_lines.append(np.array2string(np.array(test_metrics["confusion_matrix"])))

    with open(os.path.join(out_dir, "REPORT.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def build_summary_row(
    mode: str,
    label_features_mode: str,
    n_features: int = None,
    elapsed_time_min: float = None,
    best_cv_f1_macro: float = None,
    best_strategy: str = None,
    test_f1_macro: float = None,
    test_balanced_accuracy: float = None,
    artifacts_dir: str = None,
) -> Dict[str, Any]:
    """Build a summary row for results tracking."""
    return {
        "feature_set": mode,
        "label_features_mode": label_features_mode,
        "n_features": n_features,
        "elapsed_time_min": elapsed_time_min,
        "best_cv_f1_macro": best_cv_f1_macro,
        "best_strategy": best_strategy,
        "test_f1_macro": test_f1_macro,
        "test_balanced_accuracy": test_balanced_accuracy,
        "artifacts_dir": artifacts_dir,
    }
