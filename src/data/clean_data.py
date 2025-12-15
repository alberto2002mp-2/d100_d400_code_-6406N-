from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from data.load_data import load_dataframe
df = load_dataframe()

def add_0_wip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in 'wip' with 0.
    """
    if "wip" not in df.columns:
        raise KeyError("Column 'wip' not found in dataframe.")

    out = df.copy()
    out["wip"] = out["wip"].fillna(0)
    return out


import pandas as pd


def quarter_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'quarter' values from strings
    ('Quarter1', 'Quarter2', 'Quarter3', 'Quarter4', 'Quarter4')
    to integers 1, 2, 3, 4, 5.
    """
    if "quarter" not in df.columns:
        raise KeyError("Column 'quarter' not found in dataframe.")

    out = df.copy()

    mapping = {
        "Quarter1": 1,
        "Quarter2": 2,
        "Quarter3": 3,
        "Quarter4": 4,
        "Quarter5": 5,
    }

    out["quarter"] = out["quarter"].map(mapping)

    if out["quarter"].isna().any():
        bad = df.loc[out["quarter"].isna(), "quarter"].unique().tolist()
        raise ValueError(
            f"Unexpected values found in 'quarter' column: {bad}"
        )

    return out


def department_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 'department' to numeric codes:
    - sewing / sweing -> 1
    - finishing -> 2
    """
    if "department" not in df.columns:
        raise KeyError("Column 'department' not found in dataframe.")

    out = df.copy()
    s = out["department"].astype(str).str.strip().str.lower()

    mapping = {
        "sewing": 1,
        "sweing": 1,      # common misspelling in this dataset
        "finishing": 2,
    }

    out["department_numeric"] = s.map(mapping)

    if out["department_numeric"].isna().any():
        bad = out.loc[out["department_numeric"].isna(), "department"].unique().tolist()
        raise ValueError(
            f"Unexpected 'department' values found (cannot map). Values: {bad}"
        )

    return out

def fix_department_spelling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct misspelled entries in the 'department' column:
    changes 'sweing' to 'sewing'.
    """
    if "department" not in df.columns:
        raise KeyError("Column 'department' not found in dataframe.")

    out = df.copy()
    out["department"] = out["department"].replace({"sweing": "sewing"})
    return out

def day_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 'day' to numeric codes Monday=1 ... Sunday=7.
    """
    if "day" not in df.columns:
        raise KeyError("Column 'day' not found in dataframe.")

    out = df.copy()
    s = out["day"].astype(str).str.strip().str.lower()

    mapping = {
        "monday": 1,
        "tuesday": 2,
        "wednesday": 3,
        "thursday": 4,
        "friday": 5,
        "saturday": 6,
        "sunday": 7,
    }

    out["day_numeric"] = s.map(mapping)

    if out["day_numeric"].isna().any():
        bad = out.loc[out["day_numeric"].isna(), "day"].unique().tolist()
        raise ValueError(f"Unexpected 'day' values found (cannot map). Values: {bad}")

    # If you want to REPLACE original 'day', do:
    # out["day"] = out["day_numeric"]
    # out = out.drop(columns=["day_numeric"])

    return out


def remove_0_07(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where targeted_productivity == 0.07.

    Uses np.isclose to avoid float equality issues.
    """
    if "targeted_productivity" not in df.columns:
        raise KeyError("Column 'targeted_productivity' not found in dataframe.")

    out = df.copy()
    mask = np.isclose(out["targeted_productivity"].astype(float), 0.07, atol=1e-12)
    out = out.loc[~mask].reset_index(drop=True)
    return out


def date_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'date' column from string/object to pandas datetime.
    """
    if "date" not in df.columns:
        raise KeyError("Column 'date' not found in dataframe.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if out["date"].isna().any():
        bad = df.loc[out["date"].isna(), "date"].head(10).tolist()
        raise ValueError(f"Some 'date' values could not be parsed. Examples: {bad}")

    return out



def clean_all_to_df_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps in a consistent order and return df_clean.
    """
    df_clean = df.copy()
    df_clean = add_0_wip(df_clean)
    df_clean = quarter_numeric(df_clean)
    df_clean = fix_department_spelling(df_clean)
    df_clean = department_numeric(df_clean)
    df_clean = day_numeric(df_clean)
    df_clean = remove_0_07(df_clean)
    df_clean = date_datetime(df_clean)
    
    return df_clean


def save_df_clean_as_parquet(
    df_clean: pd.DataFrame,
    filepath: str | Path = Path("package/data/processed/df_clean.parquet"),
) -> Path:
    """
    Save df_clean as a parquet file. Returns the resolved path written.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(path, index=False)
    return path.resolve()

