#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random Forest classifier for PR_lifetime (multiclass) with:
- Two feature sets: PRE and POST (explicit lists)
- Balancing strategies (scikit-learn): none, undersample, oversample
- Hyperparameter tuning via GridSearchCV
- Organized artifacts + global report
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from joblib import dump

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

TARGET_COL = "PR_lifetime"
LABEL_PREFIX = "CONT_label_"

def oversample_multiclass(
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Oversample minority classes up to the maximum class size."""
    y_ser = pd.Series(y).reset_index(drop=True)
    X_df = X.reset_index(drop=True)

    counts = y_ser.value_counts()
    max_n = int(counts.max())

    X_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []

    for cls, n in counts.items():
        idx = y_ser[y_ser == cls].index
        X_cls = X_df.loc[idx]
        y_cls = y_ser.loc[idx]

        if int(n) < max_n:
            X_up, y_up = resample(
                X_cls, y_cls,
                replace=True,
                n_samples=max_n,
                random_state=random_state
            )
            X_parts.append(X_up)
            y_parts.append(y_up)
        else:
            X_parts.append(X_cls)
            y_parts.append(y_cls)

    X_bal = pd.concat(X_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    y_bal = pd.concat(y_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True).to_numpy()
    return X_bal, y_bal


def undersample_multiclass(
    X: pd.DataFrame,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Undersample majority classes down to the minimum class size."""
    y_ser = pd.Series(y).reset_index(drop=True)
    X_df = X.reset_index(drop=True)

    counts = y_ser.value_counts()
    min_n = int(counts.min())

    X_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []

    for cls, n in counts.items():
        idx = y_ser[y_ser == cls].index
        X_cls = X_df.loc[idx]
        y_cls = y_ser.loc[idx]

        if int(n) > min_n:
            X_down, y_down = resample(
                X_cls, y_cls,
                replace=False,
                n_samples=min_n,
                random_state=random_state
            )
            X_parts.append(X_down)
            y_parts.append(y_down)
        else:
            X_parts.append(X_cls)
            y_parts.append(y_cls)

    X_bal = pd.concat(X_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    y_bal = pd.concat(y_parts).sample(frac=1.0, random_state=random_state).reset_index(drop=True).to_numpy()
    return X_bal, y_bal


class ResampledRF(BaseEstimator, ClassifierMixin):
    """
    RandomForest wrapper that applies resampling inside fit(),
    ensuring resampling occurs per CV split (prevents leakage).

    Note: This wrapper expects to receive the *transformed* feature matrix
    from the preprocessing step. We convert it to a DataFrame for resampling.
    """

    def __init__(
        self,
        strategy: str = "none",
        random_state: int = 42,
        rf_params: Dict[str, Any] | None = None
    ):
        self.strategy = strategy
        self.random_state = random_state
        self.rf_params = rf_params if rf_params is not None else {}

    def fit(self, X, y):
        y_in = np.asarray(y)

        # Convert transformed matrix to DataFrame for safe row resampling
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
            **self.rf_params
        )
        self.model_.fit(X_df, y_in)
        return self

    def predict(self, X):
        return self.model_.predict(pd.DataFrame(X))


# ----------------------------
# Feature selection (explicit)
# ----------------------------

def select_features(
    df: pd.DataFrame,
    mode: str,
    pre_cols: List[str],
    post_only_cols: List[str]
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    mode:
      - 'pre' : PRE + all CONT_label_*
      - 'post': PRE + POST_ONLY + all CONT_label_*
    """
    label_cols = [c for c in df.columns if c.startswith(LABEL_PREFIX)]

    if mode == "pre":
        wanted = pre_cols + label_cols
    elif mode == "post":
        wanted = pre_cols + post_only_cols + label_cols
    else:
        raise ValueError("mode must be 'pre' or 'post'")

    used = [c for c in wanted if c in df.columns]
    X = df[used].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y, used


# ----------------------------
# Preprocess
# ----------------------------

def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


# ----------------------------
# Grid + evaluation + export
# ----------------------------

def get_param_grid() -> Dict[str, Any]:
    """
    Compact grid to keep runtime reasonable.
    You can expand later once the pipeline is stable.
    """
    return {
        "clf__strategy": ["none", "undersample", "oversample"],
        "clf__rf_params": [
            {"n_estimators": 200, "max_depth": None, "max_features": "sqrt",
             "min_samples_split": 2, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": None, "max_features": "sqrt",
             "min_samples_split": 2, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": 20, "max_features": "sqrt",
             "min_samples_split": 2, "min_samples_leaf": 1},
            {"n_estimators": 500, "max_depth": 20, "max_features": "log2",
             "min_samples_split": 10, "min_samples_leaf": 4},
        ],
    }


def best_per_strategy_from_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a summary table with the best row per balancing strategy.
    Uses mean_test_score descending (best first).
    """
    if "param_clf__strategy" not in cv_results.columns:
        return pd.DataFrame()

    cols = [
        "param_clf__strategy",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "param_clf__rf_params",
    ]
    cols = [c for c in cols if c in cv_results.columns]
    compact = cv_results[cols].copy()

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
    best_params: Dict[str, Any],
    best_cv_score: float,
    test_metrics: Dict[str, Any],
    features_used: List[str],
    cv_results: pd.DataFrame,
    best_per_strategy: pd.DataFrame,
    model: Pipeline
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Save feature list
    pd.Series(features_used, name="feature").to_csv(os.path.join(out_dir, "features_used.csv"), index=False)

    # Save CV results
    cv_results.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)

    # Save best per strategy
    best_per_strategy.to_csv(os.path.join(out_dir, "cv_best_per_strategy.csv"), index=False)

    # Save model
    dump(model, os.path.join(out_dir, "best_model.joblib"))

    # Save JSON summary
    payload = {
        "feature_set": mode,
        "n_features": len(features_used),
        "best_cv_f1_macro": best_cv_score,
        "best_params": best_params,
        "test_metrics": {
            "test_f1_macro": test_metrics["test_f1_macro"],
            "test_balanced_accuracy": test_metrics["test_balanced_accuracy"],
        },
        "artifacts_dir": out_dir,
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Save readable report
    report_lines = []
    report_lines.append(f"Feature set: {mode}")
    report_lines.append(f"Number of features: {len(features_used)}")
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


def evaluate_rf_pre_post(
    data_path: str,
    pre_cols: List[str],
    post_only_cols: List[str],
    results_root: str = "outputs_rf",
    random_state: int = 42,
    test_size: float = 0.2,
    cv_splits: int = 5,
) -> None:
    os.makedirs(results_root, exist_ok=True)

    df = pd.read_csv(data_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {data_path}")

    # Fixed stratified split for fair comparison PRE vs POST
    y_all = df[TARGET_COL].astype(int).to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, y_all))

    summary_rows: List[Dict[str, Any]] = []

    for mode in ["pre", "post"]:
        X, y, used_cols = select_features(df, mode, pre_cols, post_only_cols)

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

        param_grid = get_param_grid()
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=2,
            refit=True,
        )
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_
        best_params = gs.best_params_
        best_cv = float(gs.best_score_)

        y_pred = best_model.predict(X_test)

        test_metrics = {
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred, digits=4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        cv_results = pd.DataFrame(gs.cv_results_)
        best_strategy_df = best_per_strategy_from_cv_results(cv_results)

        exp_dir = os.path.join(results_root, mode)
        export_experiment_artifacts(
            out_dir=exp_dir,
            mode=mode,
            best_params=best_params,
            best_cv_score=best_cv,
            test_metrics=test_metrics,
            features_used=used_cols,
            cv_results=cv_results,
            best_per_strategy=best_strategy_df,
            model=best_model
        )

        summary_rows.append({
            "feature_set": mode,
            "n_features": len(used_cols),
            "best_cv_f1_macro": best_cv,
            "best_strategy": best_params.get("clf__strategy", None),
            "test_f1_macro": test_metrics["test_f1_macro"],
            "test_balanced_accuracy": test_metrics["test_balanced_accuracy"],
            "artifacts_dir": exp_dir,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("test_f1_macro", ascending=False)
    summary_df.to_csv(os.path.join(results_root, "SUMMARY.csv"), index=False)

    with open(os.path.join(results_root, "REPORT_GLOBAL.txt"), "w", encoding="utf-8") as f:
        f.write("Random Forest Experiments (PRE vs POST)\n\n")
        f.write(f"Dataset: {data_path}\n")
        f.write(f"Target: {TARGET_COL}\n")
        f.write(f"CV splits: {cv_splits}\n")
        f.write(f"Test size: {test_size}\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n")


if __name__ == "__main__":
    DATA_PATH = "../data/PREPROCESSED/Final_TtST_Dataset.csv"

    PRE_COLS = [
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

    POST_ONLY_COLS = [
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

    evaluate_rf_pre_post(
        data_path=DATA_PATH,
        pre_cols=PRE_COLS,
        post_only_cols=POST_ONLY_COLS,
        results_root="outputs_rf",
        random_state=42,
        test_size=0.2,
        cv_splits=5,
    )
