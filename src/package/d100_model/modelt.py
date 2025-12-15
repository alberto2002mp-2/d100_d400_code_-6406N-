import pandas as pd
df1 = pd.read_parquet("package/data/processed/df_clean.parquet")

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
TARGET = "actual_productivity"



def set_random_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_random_numeric(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if target not in numeric_df.columns:
        raise KeyError(f"Target '{target}' must be present and numeric.")

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    return train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)


def create_numeric_preprocessor(scale: bool) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=steps)

    return ColumnTransformer(
        transformers=[("num", numeric_transformer, slice(0, None))],
        remainder="drop",
    )


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = root_mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_tune(
    model_name: str,
    pipeline: Pipeline,
    param_dist: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 20,
) -> Pipeline:
    print(f"\n=== {model_name} ===")

    # Baseline fit
    pipeline.fit(X_train, y_train)
    base_pred = pipeline.predict(X_test)
    base_metrics = evaluate_regression(y_test, base_pred)
    print("Baseline:", base_metrics)

    # CV tuning
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    tuned_pred = search.best_estimator_.predict(X_test)
    tuned_metrics = evaluate_regression(y_test, tuned_pred)
    print("Tuned:", tuned_metrics)
    print("Best params:", search.best_params_)

    return search.best_estimator_


def run_training(df: pd.DataFrame, output_dir: Path = Path("artifacts")) -> None:
    set_random_seeds(RANDOM_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = split_random_numeric(df)

    # GLM-style baseline (linear, needs scaling)
    glm_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_numeric_preprocessor(scale=True)),
            ("model", Ridge(random_state=RANDOM_SEED)),
        ]
    )
    glm_best = train_and_tune(
        "GLM (Ridge)",
        glm_pipeline,
        param_dist={"model__alpha": loguniform(1e-4, 100)},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_iter=30,
    )

    # LGBM (no scaling)
    lgbm_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_numeric_preprocessor(scale=False)),
            ("model", LGBMRegressor(random_state=RANDOM_SEED, verbose=-1)),
        ]
    )
    lgbm_best = train_and_tune(
        "LGBM",
        lgbm_pipeline,
        param_dist={
            "model__n_estimators": randint(200, 1500),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__num_leaves": randint(10, 80),
            "model__min_child_samples": randint(5, 80),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_iter=25,
    )

    # Save artifacts
    joblib.dump(glm_best, output_dir / "glm_model.joblib")
    joblib.dump(lgbm_best, output_dir / "lgbm_model.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, output_dir / "test_data.joblib")

run_training(df1)