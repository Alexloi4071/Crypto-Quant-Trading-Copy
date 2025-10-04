# -*- coding: utf-8 -*-
"""
I/O helpers for Optuna layered system.

Provides simple, dependency-tolerant readers/writers for DataFrame and JSON,
plus an MD5 helper used by the coordinator.
"""
from __future__ import annotations

import json
import os
from hashlib import md5
from pathlib import Path
from typing import Any

import pandas as pd


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_dataframe(df: pd.DataFrame, path: os.PathLike | str) -> tuple[Path, str]:
    """Write a DataFrame to path by extension and return (actual_path, format).

    Supports .parquet, .pkl/.pickle, .csv. Creates parent directories.
    """
    target = Path(path)
    _ensure_parent(target)
    ext = target.suffix.lower()
    fmt = "parquet"
    if ext == ".parquet":
        try:
            # Prefer pyarrow if available, otherwise let pandas decide
            df.to_parquet(target, index=True)
            fmt = "parquet"
            return target, fmt
        except Exception:
            # Fallback to pickle if parquet engine not available
            alt = target.with_suffix(".pkl")
            df.to_pickle(alt)
            fmt = "pickle"
            return alt, fmt
    elif ext in (".pkl", ".pickle"):
        df.to_pickle(target)
        fmt = "pickle"
        return target, fmt
    elif ext == ".csv":
        df.to_csv(target)
        fmt = "csv"
        return target, fmt
    else:
        # Default to parquet
        df.to_parquet(target, index=True)
        fmt = "parquet"
        return target, fmt


def read_dataframe(path: os.PathLike | str) -> pd.DataFrame:
    """Read a DataFrame from path by extension.

    Supports .parquet, .pkl/.pickle, .csv.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    ext = target.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(target)
    if ext in (".pkl", ".pickle"):
        return pd.read_pickle(target)
    if ext == ".csv":
        return pd.read_csv(target, index_col=0, parse_dates=True)
    # Fallback: try parquet then pickle
    try:
        return pd.read_parquet(target)
    except Exception:
        return pd.read_pickle(target)


def compute_file_md5(path: os.PathLike | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 of a file in chunks to avoid high memory usage."""
    target = Path(path)
    hash_md5 = md5()
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def atomic_write_json(path: os.PathLike | str, data: Any, *, encoding: str = "utf-8") -> Path:
    """Atomically write JSON to a file: write to temp and replace.

    Ensures parent directories exist. Uses utf-8 and ensure_ascii=False.
    """
    target = Path(path)
    _ensure_parent(target)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding=encoding) as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(target)
    return target


