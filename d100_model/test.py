import pandas as pd

df = pd.read_parquet("data/processed/df_clean.parquet")
print(df.head())