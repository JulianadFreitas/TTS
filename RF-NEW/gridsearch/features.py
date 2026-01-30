#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature selection and ablation utilities."""

from typing import List, Tuple

import numpy as np
import pandas as pd

# Constants
TARGET_COL = "PR_lifetime"
LABEL_PREFIX = "CONT_label_"
LABEL_COUNT_COL = "CONT_label_count"


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
