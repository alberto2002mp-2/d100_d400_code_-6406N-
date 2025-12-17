from __future__ import annotations

"""Compare Ridge, Lasso, and ElasticNet regressors under a unified pipeline."""

import logging
from typing import Any, Dict, List

import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from package.models.model_training import evaluate_regression
from package.version_check import ensure_sklearn_version

logger = logging.getLogger(__name__)


def _build_preprocess_model(model: Any) -> Pipeline:
    """Create a pipeline with median imputation, scaling, then the model."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def compare_glm_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int = 42,
    n_iter: int = 20,
) -> pd.DataFrame:
    """Train/tune Ridge, Lasso, and ElasticNet with shared preprocessing/CV and return metrics.

    Returns:
        DataFrame with rows = model names and columns = rmse, mae, r2, best_params.
    """
    ensure_sklearn_version()

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = "neg_root_mean_squared_error"

    configs: Dict[str, Dict[str, Any]] = {
        "ridge": {
            "estimator": _build_preprocess_model(Ridge(random_state=None)),
            "param_distributions": {"model__alpha": loguniform(1e-4, 100)},
        },
        "lasso": {
            "estimator": _build_preprocess_model(Lasso(max_iter=10000, random_state=seed)),
            "param_distributions": {"model__alpha": loguniform(1e-4, 1)},
        },
        "elasticnet": {
            "estimator": _build_preprocess_model(
                ElasticNet(max_iter=10000, random_state=seed)
            ),
            "param_distributions": {
                "model__alpha": loguniform(1e-4, 1),
                "model__l1_ratio": uniform(0, 1),
            },
        },
    }

    rows: List[Dict[str, Any]] = []

    for name, cfg in configs.items():
        logger.info("Tuning %s model", name)
        search = RandomizedSearchCV(
            estimator=cfg["estimator"],
            param_distributions=cfg["param_distributions"],
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=seed,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        preds = best_est.predict(X_test)
        metrics = evaluate_regression(y_test, preds)
        rows.append(
            {
                "model": name,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "best_params": search.best_params_,
            }
        )

    results_df = pd.DataFrame(rows).set_index("model")
    return results_df


__all__ = ["compare_glm_models"]
