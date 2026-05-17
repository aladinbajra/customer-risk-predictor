from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.risk_metrics import predict_probabilities
    from src.customer_features import ID_COLUMN, TARGET_COLUMN
    from src.project_io import ensure_parent, project_path, setup_logging
    from src.customer_schema import validate_file_exists, validate_inference_data, validate_thresholds
except ModuleNotFoundError:
    from risk_metrics import predict_probabilities
    from customer_features import ID_COLUMN, TARGET_COLUMN
    from project_io import ensure_parent, project_path, setup_logging
    from customer_schema import validate_file_exists, validate_inference_data, validate_thresholds


LOGGER = logging.getLogger(__name__)


ACTION_MAP = {
    "low": "monitor normally",
    "medium": "proactive follow-up",
    "high": "urgent retention/escalation review",
}


DEFAULT_MEDIUM_THRESHOLD = 0.35
DEFAULT_HIGH_THRESHOLD = 0.65


def assign_risk_category(
    probability: float,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> str:
    validate_thresholds(medium_threshold, high_threshold)
    if probability >= high_threshold:
        return "high"
    if probability >= medium_threshold:
        return "medium"
    return "low"


def score_customers(
    model_path: str | Path,
    input_path: str | Path,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> pd.DataFrame:
    validate_thresholds(medium_threshold, high_threshold)
    model_file = project_path(model_path)
    input_file = project_path(input_path)
    validate_file_exists(model_file, "Production model")
    validate_file_exists(input_file, "Inference input")

    model = joblib.load(model_file)
    df = pd.read_csv(input_file)
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])
    validate_inference_data(df)
    if ID_COLUMN not in df.columns:
        df.insert(0, ID_COLUMN, [f"UPLOAD-{i:06d}" for i in range(1, len(df) + 1)])

    probabilities = predict_probabilities(model, df)
    results = df.copy()
    results["risk_probability"] = probabilities.round(6)
    results["risk_category"] = results["risk_probability"].apply(
        assign_risk_category,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    results["recommended_action"] = results["risk_category"].map(ACTION_MAP)

    lead_columns = [ID_COLUMN, "risk_probability", "risk_category", "recommended_action"]
    ordered_columns = lead_columns + [column for column in results.columns if column not in lead_columns]
    return results[ordered_columns]


def run_score_accounts(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path = "outputs/model_registry/production/customer_risk_pipeline.joblib",
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> pd.DataFrame:
    predictions = score_customers(
        model_path=model_path,
        input_path=input_path,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    output_file = ensure_parent(output_path)
    predictions.to_csv(output_file, index=False)
    LOGGER.info("Saved %s predictions to %s", len(predictions), output_file)
    return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch risk scoring for new customer accounts.")
    parser.add_argument("--input", default="data/inference/weekly_scoring_batch.csv", help="Input CSV with unlabeled accounts.")
    parser.add_argument("--output", default="outputs/predictions/customer_risk_predictions.csv", help="Output predictions CSV.")
    parser.add_argument(
        "--model",
        default="outputs/model_registry/production/customer_risk_pipeline.joblib",
        help="Path to production model artifact.",
    )
    parser.add_argument("--medium-threshold", type=float, default=DEFAULT_MEDIUM_THRESHOLD)
    parser.add_argument("--high-threshold", type=float, default=DEFAULT_HIGH_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_score_accounts(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        medium_threshold=args.medium_threshold,
        high_threshold=args.high_threshold,
    )


if __name__ == "__main__":
    main()



