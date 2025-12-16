"""Helper to build the cleaned parquet from the bundled garments CSV."""

from pathlib import Path

from data.clean_data import clean_all_to_df_clean, save_df_clean_as_parquet
from data.load_data import load_local_productivity_dataframe


def build_clean_parquet() -> Path:
    """Load the local CSV, clean it, and write the parquet. Returns the path."""
    df = load_local_productivity_dataframe()
    df_clean = clean_all_to_df_clean(df)
    return save_df_clean_as_parquet(df_clean)


if __name__ == "__main__":
    parquet_path = build_clean_parquet()
    print("Saved cleaned parquet to:", parquet_path)
