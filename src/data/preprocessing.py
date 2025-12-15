import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.load_data import load_dataframe

df = load_dataframe()

df


def plot_mean_actual_productivity_by_team(df: pd.DataFrame) -> pd.Series:
    """
    Groups the dataframe by the integer-valued `team` column (1–12),
    computes the mean of `actual_productivity` for each team,
    and displays the results as a bar chart using matplotlib.

    Returns the resulting Series.
    """
    # --- Validation ---
    if "team" not in df.columns:
        raise KeyError("Column 'team' not found in dataframe.")
    if "actual_productivity" not in df.columns:
        raise KeyError("Column 'actual_productivity' not found in dataframe.")

    # Ensure team is treated as integer groups
    df = df.copy()
    df["team"] = df["team"].astype(int)

    # --- Grouping and mean calculation ---
    mean_productivity_by_team = (
        df.groupby("team")["actual_productivity"]
        .mean()
        .sort_index()
    )

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.bar(
        mean_productivity_by_team.index,
        mean_productivity_by_team.values,
    )

    plt.xlabel("Team Number")
    plt.ylabel("Mean Actual Productivity")
    plt.title("Mean Actual Productivity by Team")
    plt.xticks(mean_productivity_by_team.index)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    return mean_productivity_by_team


def plot_mean_targeted_productivity_by_team(df: pd.DataFrame) -> pd.Series:
    """
    Groups the dataframe by the integer-valued `team` column (1–12),
    computes the mean of `targeted_productivity` for each team,
    and displays the results as a bar chart using matplotlib.

    Returns the resulting Series.
    """
    # --- Validation ---
    if "team" not in df.columns:
        raise KeyError("Column 'team' not found in dataframe.")
    if "targeted_productivity" not in df.columns:
        raise KeyError("Column 'targeted_productivity' not found in dataframe.")

    # Ensure team is treated as integer groups
    df = df.copy()
    df["team"] = df["team"].astype(int)

    # --- Grouping and mean calculation ---
    mean_productivity_by_team = (
        df.groupby("team")["targeted_productivity"]
        .mean()
        .sort_index()
    )

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.bar(
        mean_productivity_by_team.index,
        mean_productivity_by_team.values,
    )

    plt.xlabel("Team Number")
    plt.ylabel("Mean targeted Productivity")
    plt.title("Mean targeted Productivity by Team")
    plt.xticks(mean_productivity_by_team.index)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    return mean_productivity_by_team

