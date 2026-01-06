from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def load_excel(path: str | Path) -> pd.DataFrame:
    """Load an Excel file into a DataFrame."""
    return pd.read_excel(Path(path))


def to_resampled_mean(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    frequency: str = "QE",
) -> pd.DataFrame:
    """Resample a DataFrame by date column and take the mean."""
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(date_col).resample(frequency).mean().reset_index()
    return data


def select_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame with selected columns, preserving order."""
    return df.loc[:, list(columns)]


def to_numpy(df: pd.DataFrame, columns: Iterable[str] | None = None):
    """Return a numpy array from a DataFrame, optionally selecting columns."""
    if columns is None:
        return df.to_numpy()
    return df.loc[:, list(columns)].to_numpy()
