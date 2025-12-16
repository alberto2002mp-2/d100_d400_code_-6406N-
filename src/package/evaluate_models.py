from __future__ import annotations

"""Evaluate saved regression models on the persisted test split."""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from package.models.model_training import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARQUET_PATH,
    DEFAULT_SEED,
    TARGET_COLUMN,
    evaluate_regression,
    load_df_clean,
    split_random_numeric,
)

logger = logging.getLogger(__name__)


def load_clean_and_split(
    parquet_path: str | Path = DEFAULT_PARQUET_PATH, seed: int = DEFAULT_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the cleaned dataset and reproduce the training split."""
    df = load_df_clean(parquet_path)
    return split_random_numeric(df, target=TARGET_COLUMN, seed=seed)


def load_models_and_test_data(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load persisted models and the saved test split."""
    output_path = Path(output_dir)
    glm_model = joblib.load(output_path / "glm_model.joblib")
    lgbm_model = joblib.load(output_path / "lgbm_model.joblib")
    test_data = joblib.load(output_path / "test_data.joblib")

    if not isinstance(test_data, dict) or "X_test" not in test_data or "y_test" not in test_data:
        raise ValueError("test_data.joblib must contain 'X_test' and 'y_test'.")

    return glm_model, lgbm_model, test_data


def format_metrics(metrics: Dict[str, float]) -> str:
    """Return a compact string for RMSE, MAE, and R^2."""
    return (
        f"RMSE: {metrics['rmse']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"R^2: {metrics['r2']:.4f}"
    )


def plot_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
):
    """Create and save a predicted-vs-actual scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.5)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal 45deg")

    ax.set_xlabel("Actual productivity")
    ax.set_ylabel("Predicted productivity")
    ax.set_title(f"{model_name} predictions vs actuals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / f"{model_name}_pred_vs_actual.png"

    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    return fig, plot_path


def evaluate_saved_models(
    parquet_path: str | Path = DEFAULT_PARQUET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Dict[str, Any]]:
    """Load data, models, and report metrics plus plots.

    Returns a dictionary keyed by model name with metrics, predictions, and plot paths
    so the function can be reused from notebooks.
    """
    _, X_test_split, _, y_test_split = load_clean_and_split(parquet_path, seed)
    glm_model, lgbm_model, test_data = load_models_and_test_data(output_dir)

    X_test = test_data.get("X_test", X_test_split)
    y_test = test_data.get("y_test", y_test_split)

    if len(y_test) != len(y_test_split):
        logger.info(
            "Saved test set length (%s) differs from split length (%s)",
            len(y_test),
            len(y_test_split),
        )

    glm_pred = glm_model.predict(X_test)
    lgbm_pred = lgbm_model.predict(X_test)

    glm_metrics = evaluate_regression(y_test, glm_pred)
    lgbm_metrics = evaluate_regression(y_test, lgbm_pred)

    glm_fig, glm_plot_path = plot_predictions(
        pd.Series(y_test, name="actual"), pd.Series(glm_pred, name="glm_pred"), "glm", Path(output_dir) / "evaluation"
    )
    lgbm_fig, lgbm_plot_path = plot_predictions(
        pd.Series(y_test, name="actual"), pd.Series(lgbm_pred, name="lgbm_pred"), "lgbm", Path(output_dir) / "evaluation"
    )

    results = {
        "glm": {
            "metrics": glm_metrics,
            "predictions": glm_pred,
            "plot_path": glm_plot_path,
            "figure": glm_fig,
        },
        "lgbm": {
            "metrics": lgbm_metrics,
            "predictions": lgbm_pred,
            "plot_path": lgbm_plot_path,
            "figure": lgbm_fig,
        },
        "y_test": y_test,
    }

    print("ElasticNet", format_metrics(glm_metrics))
    print("LightGBM", format_metrics(lgbm_metrics))

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    evaluate_saved_models()


if __name__ == "__main__":
    main()
