"""Utilities for loading the garments worker productivity dataset.

The bundled CSV is treated as the primary source. A UCI stock dataset
fetch (id=597) remains available as a fallback or for experimentation.
All loaders return a ``pandas.DataFrame`` so they can be reused across
notebooks or other scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

__all__ = [
    "load_data",
    "load_stocks_dataframe",
    "load_stock_dataframe",
    "load_local_productivity_dataframe",
    "load_dataframe",
]

DEFAULT_LOCAL_CSV = Path(__file__).resolve().parents[2] / "garments_worker_productivity.csv"


def load_data(prefer_local: bool = True) -> pd.DataFrame:
    """Load the garments productivity dataset, preferring the bundled CSV.

    Args:
        prefer_local: Try the bundled CSV first, then fall back to the UCI fetch
            if available.

    Returns:
        DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the local CSV is missing and no fallback succeeds.
    """

    if prefer_local:
        try:
            return load_local_productivity_dataframe()
        except FileNotFoundError:
            pass

    try:
        return load_stocks_dataframe()
    except Exception:
        if not prefer_local:
            try:
                return load_local_productivity_dataframe()
            except FileNotFoundError:
                pass

    raise FileNotFoundError(
        f"Could not load dataset: expected CSV at {DEFAULT_LOCAL_CSV} or a successful UCI fetch."
    )


def load_stocks_dataframe() -> pd.DataFrame:
    """Return the UCI stock dataset (id=597) as a ``pandas.DataFrame``.

    The dataset is fetched with ``ucimlrepo.fetch_ucirepo`` and the available
    feature and target columns are combined into a single DataFrame, making it
    convenient to import and reuse across notebooks.
    """

    stocks = fetch_ucirepo(id=597)

    # ``stocks.data.features`` and ``stocks.data.targets`` are already DataFrames
    # (or a Series for single targets). Combine them when targets are present so
    # callers can work with a single table.
    features = stocks.data.features
    targets = stocks.data.targets

    if targets is None or targets.empty:
        return features.copy()

    targets_df = targets if isinstance(targets, pd.DataFrame) else targets.to_frame()
    return pd.concat([features, targets_df], axis=1)


def load_stock_dataframe() -> pd.DataFrame:
    """Alias for ``load_stocks_dataframe`` to match the expected import name."""

    return load_stocks_dataframe()


def load_local_productivity_dataframe() -> pd.DataFrame:
    """Read the bundled garments worker productivity CSV relative to the repo root."""

    if not DEFAULT_LOCAL_CSV.exists():
        raise FileNotFoundError(f"Could not find bundled CSV at {DEFAULT_LOCAL_CSV}")

    return pd.read_csv(DEFAULT_LOCAL_CSV)


def load_dataframe() -> pd.DataFrame:
    """Backward-compatible loader that prefers the bundled garments CSV."""
    return load_data(prefer_local=True)


if __name__ == "__main__":
    df = load_dataframe()
    print(df.head())
