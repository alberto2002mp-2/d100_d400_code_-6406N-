from __future__ import annotations

import pandas as pd

from load_data import load_stocks_dataframe


def get_data_description(df: pd.DataFrame) -> dict:
    """
    Generates a description of the data and plots distributions.

    Parameters:
        df: The DataFrame to analyze.

    Returns:
        A dictionary containing data types and a dataframe of descriptive statistics.
    """
    # plotting.plot_distributions(df)

    return {"dtypes": df.dtypes, "description": df.describe(include="all")}


if __name__ == "__main__":
    df = load_stocks_dataframe()
    print(df)
    df.info()
    df.describe()
