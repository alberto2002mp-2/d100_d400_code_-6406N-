from data.load_data import load_dataframe
df = load_dataframe()

from data.clean_data import clean_all_to_df_clean, save_df_clean_as_parquet

df_clean = clean_all_to_df_clean(df)
saved_path = save_df_clean_as_parquet(df_clean)

print(df_clean.head())
print(saved_path)
