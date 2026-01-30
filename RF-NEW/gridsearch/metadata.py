#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metadata tracking and serialization for reproducibility."""

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional


def now_utc_iso() -> str:
    """Current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_of_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class RunMetadata:
    """Paper-friendly metadata for reproducibility."""
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
    """Save RunMetadata to JSON."""
    os.makedirs(results_root, exist_ok=True)
    path = os.path.join(results_root, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2)
