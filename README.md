# d100_d400_code_6406N
Predicting labour productivity of manufacturing workers.

A. Installation instructions: 

Anaconda should be installed in the PC. Use the provided environment to keep library versions consistent (notably scikit-learn 1.6.x used for training the saved models):

```bash
conda env update -f environment.yml --prune
conda activate d100_env
```
Avoid running the code in environments with older scikit-learn (e.g., 1.4.x), or retrain the models if you do.

## Load the data

The utilities live in `src/data`. Run commands from the repo root (or install in editable mode with `pip install -e .` so `data.*` imports work anywhere).

- Load the raw CSV bundled with the repo:
  ```python
  from data.load_data import load_local_productivity_dataframe

  df = load_local_productivity_dataframe()
  print(df.head())
  ```

- Load the stock dataset from UCI (falls back to the CSV if download fails):
  ```python
  from data.load_data import load_dataframe

  df = load_dataframe()
  print(df.head())
  ```

- Create the cleaned dataframe and write it to parquet (`src/data/clean_data/processed/df_clean.parquet` by default):
  ```python
  from data.load_data import load_local_productivity_dataframe
  from data.clean_data import clean_all_to_df_clean, save_df_clean_as_parquet

  df = load_local_productivity_dataframe()
  df_clean = clean_all_to_df_clean(df)
  parquet_path = save_df_clean_as_parquet(df_clean)
  print(parquet_path)
  ```
  (You can also run `python src/data/clean_data/processed/df_clean.parquet` to do this end to end.)

- Load the cleaned parquet back into a dataframe:
   
  ```python
    import pandas as pd
    df1 = pd.read_parquet("src/data/clean_data/processed/df_clean.parquet")
    

    # Jupter notebook
    from pathlib import Path
    import pandas as pd

    def load_df1_clean() -> pd.DataFrame:
        # Start from the file’s directory if __file__ exists, otherwise from the CWD (e.g., in notebooks)
        start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

        for base in (start, *start.parents):
            parquet_path = base / "src" / "data" / "clean_data" / "processed" / "df_clean.parquet"
            if parquet_path.exists():
                return pd.read_parquet(parquet_path)

        raise FileNotFoundError("Could not find src/data/clean_data/processed/df_clean.parquet relative to this location.")

    # Run it on any notebook:
    df1 = load_df1_clean()


B. Running code instructions:

Run any scripts or notebooks (recommended) to see the output of the code. Select d100_env as kernel

Run pytest as a bash prompt in the terminal to check that tests are passed or not 



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


