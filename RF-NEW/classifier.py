import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from joblib import dump


# ----------------------------
# Resampling (scikit-learn only)
# ----------------------------

def _to_numpy(X):
    """Ensure X is a numpy array for indexing."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.to_numpy()
    return np.asarray(X)

def _to_pandas_frame(X, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Convert to DataFrame when needed (helpful for ColumnTransformer)."""
    if isinstance(X, pd.DataFrame):
        return X
    X_np = _to_numpy(X)
    return pd.DataFrame(X_np, columns=columns)

def oversample_multiclass(X, y, random_state: int = 42):
    """
    Oversample all minority classes up to the maximum class size.
    Uses sklearn.utils.resample.
    """
    X_df = _to_pandas_frame(X)
    y_ser = pd.Series(y).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    class_counts = y_ser.value_counts()
    max_n = class_counts.max()

    X_parts = []
    y_parts = []

    for cls, n in class_counts.items():
        idx = y_ser[y_ser == cls].index
        X_cls = X_df.loc[idx]
        y_cls = y_ser.loc[idx]

        if n < max_n:
            X_up, y_up = resample(
                X_cls,
                y_cls,
                replace=True,
                n_samples=max_n,
                random_state=random_state,
            )
            X_parts.append(X_up)
            y_parts.append(y_up)
        else:
            X_parts.append(X_cls)
            y_parts.append(y_cls)

    X_bal = pd.concat(X_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    y_bal = pd.concat(y_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return X_bal, y_bal.to_numpy()

def undersample_multiclass(X, y, random_state: int = 42):
    """
    Undersample all majority classes down to the minimum class size.
    Uses sklearn.utils.resample.
    """
    X_df = _to_pandas_frame(X)
    y_ser = pd.Series(y).reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)

    class_counts = y_ser.value_counts()
    min_n = class_counts.min()

    X_parts = []
    y_parts = []

    for cls, n in class_counts.items():
        idx = y_ser[y_ser == cls].index
        X_cls = X_df.loc[idx]
        y_cls = y_ser.loc[idx]

        if n > min_n:
            X_down, y_down = resample(
                X_cls,
                y_cls,
                replace=False,
                n_samples=min_n,
                random_state=random_state,
            )
            X_parts.append(X_down)
            y_parts.append(y_down)
        else:
            X_parts.append(X_cls)
            y_parts.append(y_cls)

    X_bal = pd.concat(X_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    y_bal = pd.concat(y_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return X_bal, y_bal.to_numpy()


class ResampledClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that applies resampling INSIDE fit().
    This ensures resampling happens per CV split (avoids leakage).
    """
    def __init__(
        self,
        base_estimator=None,
        strategy: str = "none",  # "none" | "oversample" | "undersample"
        random_state: int = 42,
    ):
        self.base_estimator = base_estimator if base_estimator is not None else RandomForestClassifier()
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X, y):
        X_in = X
        y_in = np.asarray(y)

        if self.strategy == "oversample":
            X_in, y_in = oversample_multiclass(X_in, y_in, random_state=self.random_state)
        elif self.strategy == "undersample":
            X_in, y_in = undersample_multiclass(X_in, y_in, random_state=self.random_state)

        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X_in, y_in)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X)
        raise AttributeError("Underlying estimator does not support predict_proba().")


# ----------------------------
# Training / evaluation
# ----------------------------

@dataclass
class ExperimentConfig:
    data_path: str
    target_col: str = "PR_lifetime"
    test_size: float = 0.2
    random_state: int = 42
    cv_splits: int = 5
    min_occurrences_labels: int = 10  # not used here; kept for consistency
    output_dir: str = "outputs"


def build_preprocess_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def run_experiment(cfg: ExperimentConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    df = pd.read_csv(cfg.data_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in dataset.")

    y = df[cfg.target_col].astype(int).to_numpy()
    X = df.drop(columns=[cfg.target_col])

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    pre = build_preprocess_pipeline(X_train)

    # Base RF (will be tuned)
    rf = RandomForestClassifier(
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    # Wrapper to handle resampling per fit
    clf = ResampledClassifier(base_estimator=rf, strategy="none", random_state=cfg.random_state)

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("clf", clf),
        ]
    )

    # Grid (feel free to expand later)
    param_grid = {
        "clf__strategy": ["none", "undersample", "oversample"],
        "clf__base_estimator__n_estimators": [200, 500],
        "clf__base_estimator__max_depth": [None, 10, 20],
        "clf__base_estimator__min_samples_split": [2, 5, 10],
        "clf__base_estimator__min_samples_leaf": [1, 2, 4],
        "clf__base_estimator__max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.random_state)

    # Use macro-F1 to account for imbalance
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
    best_cv_score = gs.best_score_

    y_pred = best_model.predict(X_test)

    results = {
        "best_params": best_params,
        "best_cv_f1_macro": float(best_cv_score),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("\n=== BEST CONFIG ===")
    print("Best CV macro-F1:", best_cv_score)
    print("Best params:", best_params)
    print("\n=== TEST METRICS ===")
    print("Test macro-F1:", results["test_f1_macro"])
    print("Test balanced accuracy:", results["test_balanced_accuracy"])
    print("\nClassification report:\n", results["classification_report"])
    print("\nConfusion matrix:\n", np.array(results["confusion_matrix"]))

    # Save artifacts
    model_path = os.path.join(cfg.output_dir, "best_rf_model.joblib")
    dump(best_model, model_path)

    results_path = os.path.join(cfg.output_dir, "rf_gridsearch_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved model: {model_path}")
    print(f"✅ Saved results: {results_path}")


if __name__ == "__main__":
    config = ExperimentConfig(
        data_path="../data/PREPROCESSED/Final_TtST_Dataset.csv",
        target_col="PR_lifetime",
        output_dir="outputs_rf",
        cv_splits=5,
        test_size=0.2,
        random_state=42,
    )
    run_experiment(config)
