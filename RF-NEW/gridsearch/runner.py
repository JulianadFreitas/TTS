#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

warnings.filterwarnings("ignore")


# ----------------------------
# Constants
# ----------------------------
TARGET_COL = "PR_lifetime"
LABEL_PREFIX = "CONT_label_"
LABEL_COUNT_COL = "CONT_label_count"
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
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 for a file (used for reproducibility metadata)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Sampling
# ----------------------------
def resample_multiclass(
    X: pd.DataFrame,
    y: np.ndarray,
    strategy: str = "oversample",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Resample multiclass data.
    
    Args:
        X: Feature matrix
        y: Target array
        strategy: "oversample" (up to max) or "undersample" (down to min)
        random_state: Random seed for reproducibility
    
    Returns:
        Resampled (X, y) tuple
    """
    y_ser = pd.Series(y).reset_index(drop=True)
    X_df = X.reset_index(drop=True)

    counts = y_ser.value_counts()
    
    if strategy == "oversample":
        target_n = int(counts.max())
        replace = True
        condition = lambda n, target: n < target
    elif strategy == "undersample":
        target_n = int(counts.min())
        replace = False
        condition = lambda n, target: n > target
    else:
        raise ValueError("strategy must be 'oversample' or 'undersample'")

    X_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []

    for cls, n in counts.items():
        idx = y_ser[y_ser == cls].index
        X_cls = X_df.loc[idx]
        y_cls = y_ser.loc[idx]

        if condition(int(n), target_n):
            X_res, y_res = resample(
                X_cls,
                y_cls,
                replace=replace,
                n_samples=target_n,
                random_state=random_state,
            )
            X_parts.append(X_res)
            y_parts.append(y_res)
        else:
            X_parts.append(X_cls)
            y_parts.append(y_cls)

    X_bal = (
        pd.concat(X_parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    y_bal = (
        pd.concat(y_parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
        .to_numpy()
    )
    return X_bal, y_bal


def oversample_multiclass(
    X: pd.DataFrame, y: np.ndarray, random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Oversample minority classes up to the maximum class size."""
    return resample_multiclass(X, y, strategy="oversample", random_state=random_state)


def undersample_multiclass(
    X: pd.DataFrame, y: np.ndarray, random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Undersample majority classes down to the minimum class size."""
    return resample_multiclass(X, y, strategy="undersample", random_state=random_state)


class ResampledRF(BaseEstimator, ClassifierMixin):
    """RandomForest wrapper that optionally resamples data before fitting."""

    def __init__(
        self,
        strategy: str = "none",
        random_state: int = 42,
        n_estimators: int = 500,
        max_depth=None,
        max_features: Any = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Any = None,
    ):
        self.strategy = strategy
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight

    def fit(self, X, y):
        y_in = np.asarray(y)
        X_df = pd.DataFrame(X)

        if self.strategy == "oversample":
            X_df, y_in = oversample_multiclass(X_df, y_in, random_state=self.random_state)
        elif self.strategy == "undersample":
            X_df, y_in = undersample_multiclass(X_df, y_in, random_state=self.random_state)
        elif self.strategy != "none":
            raise ValueError("strategy must be one of: none, undersample, oversample")

        self.model_ = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=-1,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
        )
        self.model_.fit(X_df, y_in)
        return self

    def predict(self, X):
        return self.model_.predict(pd.DataFrame(X))


# ----------------------------
# Feature selection (ablation-ready)
# ----------------------------
def get_label_indicator_cols(df: pd.DataFrame) -> List[str]:
    """All label indicator columns except the aggregate count."""
    return [
        c for c in df.columns
        if c.startswith(LABEL_PREFIX) and c != LABEL_COUNT_COL
    ]


def select_features(
    df: pd.DataFrame,
    mode: str,
    pre_cols: List[str],
    post_only_cols: List[str],
    label_features_mode: str,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Select features for PRE/POST, optionally including label indicator features.

    label_features_mode:
      - "full": include all CONT_label_* (except CONT_label_count)
      - "count_only": exclude CONT_label_* except CONT_label_count
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    if mode not in {"pre", "post"}:
        raise ValueError("mode must be 'pre' or 'post'")

    if label_features_mode not in {"full", "count_only"}:
        raise ValueError("label_features_mode must be 'full' or 'count_only'")

    if mode == "pre":
        wanted = list(pre_cols)
    else:
        wanted = list(pre_cols) + list(post_only_cols)

    label_cols = get_label_indicator_cols(df)

    if label_features_mode == "full":
        wanted = wanted + label_cols
    else:
        # Defensive: ensure no label indicators sneak in
        wanted = [
            c for c in wanted
            if not (c.startswith(LABEL_PREFIX) and c != LABEL_COUNT_COL)
        ]

    wanted = [c for c in wanted if c in df.columns]
    used = list(dict.fromkeys(wanted))  # stable unique

    if not used:
        raise ValueError(f"No features selected for mode='{mode}'. Check feature lists and dataframe columns.")

    X = df[used].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y, used


# ----------------------------
# Preprocess
# ----------------------------
def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    """Impute missing values + one-hot encode categorical columns."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


# ----------------------------
# Parameter grid
# ----------------------------
def get_param_grid_search() -> Dict[str, Any]:
    return {
        "clf__strategy": ["none", "undersample", "oversample"],
        "clf__n_estimators": [200, 500, 1000],
        "clf__max_depth": [None, 20, 40],
        "clf__max_features": ["sqrt", "log2", 0.5],
        "clf__min_samples_split": [2, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4],
    }


def resolve_search_space() -> Dict[str, Any]:
    """
    Return the parameter grid for full GridSearchCV.
    """
    return get_param_grid_search()


# ----------------------------
# Reporting / artifacts
# ----------------------------
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
    report_lines: List[str] = []
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


@dataclass(frozen=True)
class RunMetadata:
    created_utc: str
    data_path: str
    data_sha256: Optional[str]
    target_col: str
    label_features_mode: str
    random_state: int
    test_size: float
    cv_splits: int
    modes: List[str]
    pre_cols: List[str]
    post_only_cols: List[str]
    sklearn_version: str
    python_version: str


def save_run_metadata(results_root: str, metadata: RunMetadata) -> None:
    path = os.path.join(results_root, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)


def build_summary_row(
    mode: str,
    label_features_mode: str,
    n_features: Optional[int] = None,
    elapsed_time_min: Optional[float] = None,
    best_cv_f1_macro: Optional[float] = None,
    best_strategy: Optional[str] = None,
    test_f1_macro: Optional[float] = None,
    test_balanced_accuracy: Optional[float] = None,
    artifacts_dir: Optional[str] = None,
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
        sklearn_version=__import__("sklearn").__version__,
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
    """
    Create a paper-friendly results_root name that encodes the experimental condition.
    """
    name = f"rf__labels-{label_features_mode}__seed-{random_state}"
    return os.path.join(outputs_base, name)


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
