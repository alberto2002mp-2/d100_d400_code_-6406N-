# d100_d400_code_6406N
Predicting labour productivity of manufacturing workers.

## Environment
- Create the conda env: `conda env create -f environment.yml`
- Activate it: `conda activate d100_env`
- Optional for imports anywhere: `pip install -e .`

## Data sources
- Primary raw file: `garments_worker_productivity.csv` (bundled at the repo root).
- If it is missing, download the same CSV and place it at the repo root with that exact name.
- An optional UCI stock dataset (id=597) remains available via `ucimlrepo` for experimentation.

## Loading data
- Preferred loader (uses the bundled garments CSV, then falls back to UCI if needed):
  ```python
  from data.load_data import load_data

  df = load_data()
  print(df.head())
  ```
- Explicit local read:
  ```python
  from data.load_data import load_local_productivity_dataframe

  df = load_local_productivity_dataframe()
  ```
- UCI stock dataset example:
  ```python
  from data.load_data import load_stocks_dataframe

  df_stock = load_stocks_dataframe()
  ```

## Cleaned dataset
- Build the cleaned parquet (`src/data/clean_data/processed/df_clean.parquet`) from the bundled CSV:
  ```
  python -m data.parquetfile
  ```
- Or in code:
  ```python
  from data.parquetfile import build_clean_parquet

  parquet_path = build_clean_parquet()
  print(parquet_path)
  ```
- Load the cleaned parquet:
  ```python
  import pandas as pd

  df_clean = pd.read_parquet("src/data/clean_data/processed/df_clean.parquet")
  ```

## Description of variables in dataset
01 date                   : Date in MM-DD-YYYY  
02 day                    : Day of the Week  
03 quarter                : A portion of the month. A month was divided into four quarters  
04 department             : Associated department with the instance  
05 team_no                : Associated team number with the instance  
06 no_of_workers          : Number of workers in each team  
07 no_of_style_change     : Number of changes in the style of a particular product  
08 targeted_productivity  : Targeted productivity set by the Authority for each team for each day.  
09 smv                    : Standard Minute Value, it is the allocated time for a task  
10 wip                    : Work in progress. Includes the number of unfinished items for products  
11 over_time              : Represents the amount of overtime by each team in minutes  
12 incentive              : Represents the amount of financial incentive (in BDT) that enables or motivates a particular course of action.  
13 idle_time              : The amount of time when the production was interrupted due to several reasons  
14 idle_men               : The number of workers who were idle due to production interruption  
15 actual_productivity    : The actual % of productivity that was delivered by the workers. It ranges from 0-1.
