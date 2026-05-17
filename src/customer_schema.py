from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.customer_features import CATEGORICAL_FEATURES, ID_COLUMN, RAW_NUMERIC_FEATURES, TARGET_COLUMN
except ModuleNotFoundError:
    from customer_features import CATEGORICAL_FEATURES, ID_COLUMN, RAW_NUMERIC_FEATURES, TARGET_COLUMN


REQUIRED_MODEL_INPUT_COLUMNS = tuple(RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def validate_file_exists(path: str | Path, description: str) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"{description} is not a file: {file_path}")
    return file_path


def validate_required_columns(df: pd.DataFrame, required_columns: list[str] | tuple[str, ...], context: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{context} is missing required column(s): {missing_text}")


def validate_training_data(df: pd.DataFrame, context: str = "Training data") -> None:
    if df.empty:
        raise ValueError(f"{context} is empty.")
    validate_required_columns(df, REQUIRED_MODEL_INPUT_COLUMNS + (TARGET_COLUMN,), context)
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"{context} contains missing values in target column '{TARGET_COLUMN}'.")
    labels = set(pd.Series(df[TARGET_COLUMN]).dropna().unique())
    if not labels.issubset({0, 1}) or len(labels) < 2:
        raise ValueError(f"{context} must contain binary target values 0 and 1 in '{TARGET_COLUMN}'.")


def validate_inference_data(df: pd.DataFrame, context: str = "Inference data") -> None:
    if df.empty:
        raise ValueError(f"{context} is empty.")
    validate_required_columns(df, REQUIRED_MODEL_INPUT_COLUMNS, context)


def validate_prediction_output_schema(df: pd.DataFrame, context: str = "Prediction output") -> None:
    required = (ID_COLUMN, "risk_probability", "risk_category", "recommended_action")
    validate_required_columns(df, required, context)


def validate_thresholds(medium_threshold: float, high_threshold: float) -> None:
    if not 0 <= medium_threshold < high_threshold <= 1:
        raise ValueError(
            "Risk thresholds must satisfy 0 <= medium_threshold < high_threshold <= 1. "
            f"Received medium={medium_threshold}, high={high_threshold}."
        )



