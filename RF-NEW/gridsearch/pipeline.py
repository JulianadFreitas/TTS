#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ML pipeline construction utilities."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sampling import ResampledRF


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


def build_pipeline(random_state: int = 42, strategy: str = "none") -> Pipeline:
    """Build the preprocessing + ResampledRF pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers=[], remainder="drop")),  # Will be filled during fit
            ("clf", ResampledRF(strategy=strategy, random_state=random_state)),
        ]
    )
