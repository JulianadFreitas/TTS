#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperparameter grids and search space resolution."""

from typing import Any, Dict


def get_param_grid_search() -> Dict[str, Any]:
    """Parameter grid for full GridSearchCV."""
    return {
        "clf__strategy": ["none", "undersample", "oversample"],
        "clf__n_estimators": [200, 500, 1000],
        "clf__max_depth": [None, 20, 40],
        "clf__max_features": ["sqrt", "log2", 0.5],
        "clf__min_samples_split": [2, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4],
    }


def resolve_search_space() -> Dict[str, Any]:
    """Return the parameter grid for full GridSearchCV."""
    return get_param_grid_search()
