#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resampling utilities for multiclass data balance."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample


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

    X_parts = []
    y_parts = []

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
        max_features = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight = None,
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
