#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-friendly Random Forest runner with label-features ablation.
Modularized version with imports from specialized modules.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline

# Local imports from modularized components
from features import TARGET_COL, select_features
from metadata import RunMetadata, now_utc_iso, save_run_metadata, sha256_of_file
from params import resolve_search_space
from pipeline import build_preprocess
from reporting import (
    best_per_strategy_from_cv_results,
    build_summary_row,
    ensure_dir,
    export_experiment_artifacts,
)
from sampling import ResampledRF

warnings.filterwarnings("ignore")


# ----------------------------
# Default feature lists
# ----------------------------
DEFAULT_PRE_COLS = [
    "pr_number",
    "CONT_repo",
    "CONT_label_count",
    "CONT_day_of_week",
    "CONT_weekday",
    "CONT_PR_text_wordiness",
    "CODE_number_of_commits",
    "CODE_lines_of_code_changed",
    "CODE_number_of_files_changed",
    "CODE_number_of_milestones",
    "ORG_author_PRs_opened",
    "ORG_PRs_opened_in_last_2_weeks",
    "ORG_PRs_closed_in_last_2_weeks",
    "ORG_open_PRs_at_open_date",
    "ACT_time_since_last_commit",
    "ACT_number_of_linked_issues",
]

DEFAULT_POST_ONLY_COLS = [
    "CODE_number_of_revisions",
    "ACT_number_of_comments",
    "ACT_number_of_review_comments",
    "ACT_number_of_reviewers",
    "ACT_number_of_approvals",
    "ACT_time_to_first_response",
    "ACT_number_of_assignees",
    "ACT_number_of_changes_requested",
    "ACT_number_of_reviews_requested",
]


# ----------------------------
# Utilities
# ----------------------------
def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_lists_from_json(path: str) -> Tuple[List[str], List[str]]:
    """
    Load feature lists from JSON:
      {
        "pre_cols": [...],
        "post_only_cols": [...]
      }
    """
    data = load_json(path)
    pre_cols = data.get("pre_cols", [])
    post_only_cols = data.get("post_only_cols", [])
    if not isinstance(pre_cols, list) or not isinstance(post_only_cols, list):
        raise ValueError("feature JSON must contain lists: pre_cols and post_only_cols")
    return pre_cols, post_only_cols


def auto_results_root(
    outputs_base: str,
    label_features_mode: str,
    random_state: int,
) -> str:
    """Create a paper-friendly results_root name."""
    name = f"rf__labels-{label_features_mode}__seed-{random_state}"
    return os.path.join(outputs_base, name)


# ----------------------------
# Main evaluation
# ----------------------------
def evaluate_rf_pre_post(
    data_path: str,
    pre_cols: List[str],
    post_only_cols: List[str],
    results_root: str,
    label_features_mode: str,
    random_state: int = 42,
    test_size: float = 0.2,
    cv_splits: int = 5,
    modes_to_run: Optional[List[str]] = None,
    n_jobs_search: int = 1,
    verbose: int = 2,
    save_data_hash: bool = True,
) -> None:
    """Main evaluation loop: PRE vs POST with label ablation."""
    ensure_dir(results_root)

    # Save run metadata (paper-friendly)
    data_hash = sha256_of_file(data_path) if save_data_hash else None
    metadata = RunMetadata(
        created_utc=now_utc_iso(),
        data_path=data_path,
        data_sha256=data_hash,
        target_col=TARGET_COL,
        label_features_mode=label_features_mode,
        random_state=int(random_state),
        test_size=float(test_size),
        cv_splits=int(cv_splits),
        modes=modes_to_run or ["pre", "post"],
        pre_cols=list(pre_cols),
        post_only_cols=list(post_only_cols),
        sklearn_version=sklearn.__version__,
        python_version=sys.version.replace("\n", " "),
    )
    save_run_metadata(results_root, metadata)

    print("\n" + "=" * 88)
    print("STARTING RF EXPERIMENT")
    print(f"Results root: {results_root}")
    print(f"Label features mode: {label_features_mode}")
    print(f"Search method: full (GridSearchCV)")
    print("=" * 88 + "\n")

    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {data_path}")

    # Fixed stratified split (independent of feature selection)
    y_all = df[TARGET_COL].astype(int).to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, y_all))

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search_space = resolve_search_space()

    modes = modes_to_run or ["pre", "post"]
    summary_rows: List[Dict[str, Any]] = []

    for mode in modes:
        print("\n" + "=" * 88)
        print(f"FEATURE SET: {mode.upper()}")
        print("=" * 88 + "\n")

        # Checkpoint: skip if results.json exists for this mode
        checkpoint_file = os.path.join(results_root, mode, "results.json")
        if os.path.exists(checkpoint_file):
            print(f"⚠️  Mode '{mode}' already completed. Skipping (checkpoint found).")
            prev = load_json(checkpoint_file)
            summary_rows.append(
                build_summary_row(
                    mode=mode,
                    label_features_mode=prev.get("label_features_mode", label_features_mode),
                    n_features=prev.get("n_features"),
                    elapsed_time_min=prev.get("elapsed_time_minutes"),
                    best_cv_f1_macro=prev.get("best_cv_f1_macro"),
                    best_strategy=prev.get("best_params", {}).get("clf__strategy"),
                    test_f1_macro=prev.get("test_metrics", {}).get("test_f1_macro"),
                    test_balanced_accuracy=prev.get("test_metrics", {}).get("test_balanced_accuracy"),
                    artifacts_dir=os.path.join(results_root, mode),
                )
            )
            continue

        start_time = time.time()

        X, y, used_cols = select_features(
            df=df,
            mode=mode,
            pre_cols=pre_cols,
            post_only_cols=post_only_cols,
            label_features_mode=label_features_mode,
        )

        print(f"Selected features: {len(used_cols)}")
        print(f"Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y[test_idx]

        preprocess = build_preprocess(X_train)

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("clf", ResampledRF(strategy="none", random_state=random_state)),
            ]
        )

        searcher = GridSearchCV(
            estimator=pipe,
            param_grid=search_space,
            scoring="f1_macro",
            cv=cv,
            n_jobs=int(n_jobs_search),
            verbose=int(verbose),
            refit=True,
        )

        print("\nStarting hyperparameter search...")
        searcher.fit(X_train, y_train)

        elapsed = time.time() - start_time
        print(f"\n✓ Search completed in {elapsed/60:.2f} minutes")

        best_model = searcher.best_estimator_
        best_params = searcher.best_params_
        best_cv = float(searcher.best_score_)

        print(f"Best CV macro-F1: {best_cv:.4f}")
        print(f"Best params: {best_params}\n")

        y_pred = best_model.predict(X_test)

        test_metrics = {
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, digits=4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        print(f"Test macro-F1: {test_metrics['test_f1_macro']:.4f}")
        print(f"Test balanced acc: {test_metrics['test_balanced_accuracy']:.4f}")

        cv_results = pd.DataFrame(searcher.cv_results_)
        best_strategy_df = best_per_strategy_from_cv_results(cv_results)

        exp_dir = os.path.join(results_root, mode)
        export_experiment_artifacts(
            out_dir=exp_dir,
            mode=mode,
            label_features_mode=label_features_mode,
            best_params=best_params,
            best_cv_score=best_cv,
            test_metrics=test_metrics,
            features_used=used_cols,
            cv_results=cv_results,
            best_per_strategy=best_strategy_df,
            model=best_model,
            elapsed_time=elapsed,
        )

        summary_rows.append(
            build_summary_row(
                mode=mode,
                label_features_mode=label_features_mode,
                n_features=int(len(used_cols)),
                elapsed_time_min=round(elapsed / 60, 2),
                best_cv_f1_macro=best_cv,
                best_strategy=best_params.get("clf__strategy"),
                test_f1_macro=test_metrics["test_f1_macro"],
                test_balanced_accuracy=test_metrics["test_balanced_accuracy"],
                artifacts_dir=exp_dir,
            )
        )

        # Free memory aggressively between modes
        del X, y, X_train, X_test, y_train, y_test
        del preprocess, pipe, searcher, best_model
        del cv_results, best_strategy_df
        gc.collect()

    # Final summary
    summary_df = pd.DataFrame(summary_rows).sort_values("test_f1_macro", ascending=False)
    summary_df.to_csv(os.path.join(results_root, "SUMMARY.csv"), index=False)

    with open(os.path.join(results_root, "REPORT_GLOBAL.txt"), "w", encoding="utf-8") as f:
        f.write("Random Forest Experiments (PRE vs POST)\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Data SHA256: {data_hash}\n")
        f.write(f"Target: {TARGET_COL}\n")
        f.write(f"Label features mode: {label_features_mode}\n")
        f.write(f"Search method: full (GridSearchCV)\n")
        f.write(f"CV splits: {cv_splits}\n")
        f.write(f"Test size: {test_size}\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")

    print("\n" + "=" * 88)
    print("EXPERIMENT FINISHED")
    print(f"Results saved to: {results_root}")
    print("=" * 88 + "\n")


# ----------------------------
# CLI / Entry point
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-friendly RF runner with label-features ablation (full vs count_only)."
    )
    parser.add_argument(
        "--data-path",
        default="../../data/PREPROCESSED/Final_TtST_Dataset.csv",
        help="Path to the preprocessed dataset CSV.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help="Output directory root. If omitted, an automatic name will be created under --outputs-base.",
    )
    parser.add_argument(
        "--outputs-base",
        default="outputs",
        help="Base directory used only when --results-root is not provided.",
    )
    parser.add_argument(
        "--label-features-mode",
        choices=["full", "count_only"],
        default="full",
        help="Ablation condition: include all label indicators (full) or keep only label_count (count_only).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["pre", "post"],
        default=["pre", "post"],
        help="Which feature sets to run.",
    )
    parser.add_argument(
        "--only-post",
        action="store_true",
        help="Convenience flag: run only POST.",
    )
    parser.add_argument(
        "--n-jobs-search",
        type=int,
        default=1,
        help="Parallel jobs for CV search. Keep low to reduce memory usage.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Verbosity for the CV search.",
    )
    parser.add_argument(
        "--feature-lists-json",
        default=None,
        help="Optional JSON file with pre_cols and post_only_cols to override defaults.",
    )
    parser.add_argument(
        "--no-data-hash",
        action="store_true",
        help="Disable SHA256 of dataset in run metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.feature_lists_json:
        pre_cols, post_only_cols = load_feature_lists_from_json(args.feature_lists_json)
    else:
        pre_cols, post_only_cols = DEFAULT_PRE_COLS, DEFAULT_POST_ONLY_COLS

    modes_to_run = ["post"] if args.only_post else list(args.modes)
    
    for label_mode in ["full", "count_only"]:
        # Avoid overwriting when user provides a custom --results-root
        results_root = args.results_root
        if results_root:
            results_root = os.path.join(results_root, f"labels-{label_mode}")
        else:
            results_root = auto_results_root(
                outputs_base=args.outputs_base,
                label_features_mode=label_mode,
                random_state=args.random_state,
            )

        evaluate_rf_pre_post(
            data_path=args.data_path,
            pre_cols=pre_cols,
            post_only_cols=post_only_cols,
            results_root=results_root,
            label_features_mode=label_mode,
            random_state=args.random_state,
            test_size=args.test_size,
            cv_splits=args.cv_splits,
            modes_to_run=modes_to_run,
            n_jobs_search=args.n_jobs_search,
            verbose=args.verbose,
            save_data_hash=not args.no_data_hash,
        )


if __name__ == "__main__":
    main()
