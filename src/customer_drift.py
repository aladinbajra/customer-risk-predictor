from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.customer_features import ID_COLUMN, TARGET_COLUMN
    from src.project_io import ensure_dir, ensure_parent, project_path, setup_logging
    from src.customer_schema import validate_file_exists, validate_inference_data, validate_prediction_output_schema, validate_training_data
except ModuleNotFoundError:
    from customer_features import ID_COLUMN, TARGET_COLUMN
    from project_io import ensure_dir, ensure_parent, project_path, setup_logging
    from customer_schema import validate_file_exists, validate_inference_data, validate_prediction_output_schema, validate_training_data


LOGGER = logging.getLogger(__name__)
EPSILON = 1e-6


def psi_from_distributions(reference_pct: np.ndarray, current_pct: np.ndarray) -> float:
    reference_pct = np.clip(reference_pct, EPSILON, 1)
    current_pct = np.clip(current_pct, EPSILON, 1)
    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference_clean = reference.dropna()
    current_clean = current.dropna()
    if reference_clean.nunique() < 2 or current_clean.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(reference_clean.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        edges = np.linspace(reference_clean.min(), reference_clean.max(), min(bins, reference_clean.nunique()) + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = pd.cut(reference_clean, bins=edges, include_lowest=True).value_counts(sort=False)
    current_counts = pd.cut(current_clean, bins=edges, include_lowest=True).value_counts(sort=False)
    return psi_from_distributions(
        reference_counts.to_numpy() / max(reference_counts.sum(), 1),
        current_counts.to_numpy() / max(current_counts.sum(), 1),
    )


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    reference_dist = reference.fillna("missing").astype(str).value_counts(normalize=True)
    current_dist = current.fillna("missing").astype(str).value_counts(normalize=True)
    categories = sorted(set(reference_dist.index).union(current_dist.index))
    return psi_from_distributions(
        np.array([reference_dist.get(category, EPSILON) for category in categories]),
        np.array([current_dist.get(category, EPSILON) for category in categories]),
    )


def drift_severity(score: float) -> str:
    if score >= 0.25:
        return "high"
    if score >= 0.10:
        return "moderate"
    return "low"


def overall_drift_status(max_psi: float) -> str:
    return drift_severity(max_psi)


def build_feature_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    common_columns = [
        column
        for column in reference_df.columns
        if column in current_df.columns and column not in {ID_COLUMN, TARGET_COLUMN}
    ]

    numeric_columns = [
        column
        for column in common_columns
        if pd.api.types.is_numeric_dtype(reference_df[column]) and pd.api.types.is_numeric_dtype(current_df[column])
    ]
    categorical_columns = [column for column in common_columns if column not in numeric_columns]

    for column in numeric_columns:
        reference_mean = reference_df[column].mean()
        current_mean = current_df[column].mean()
        absolute_change = current_mean - reference_mean
        percent_change = absolute_change / reference_mean if pd.notna(reference_mean) and reference_mean != 0 else np.nan
        psi_score = numeric_psi(reference_df[column], current_df[column])
        rows.append(
            {
                "metric_type": "numeric_mean_and_psi",
                "feature": column,
                "category": "",
                "reference_value": reference_mean,
                "current_value": current_mean,
                "absolute_change": absolute_change,
                "percent_change": percent_change,
                "psi_score": psi_score,
                "drift_severity": drift_severity(psi_score),
            }
        )

    for column in categorical_columns:
        psi_score = categorical_psi(reference_df[column], current_df[column])
        rows.append(
            {
                "metric_type": "categorical_psi",
                "feature": column,
                "category": "",
                "reference_value": "",
                "current_value": "",
                "absolute_change": "",
                "percent_change": "",
                "psi_score": psi_score,
                "drift_severity": drift_severity(psi_score),
            }
        )
        reference_dist = reference_df[column].fillna("missing").astype(str).value_counts(normalize=True)
        current_dist = current_df[column].fillna("missing").astype(str).value_counts(normalize=True)
        for category in sorted(set(reference_dist.index).union(current_dist.index)):
            reference_pct = reference_dist.get(category, 0.0)
            current_pct = current_dist.get(category, 0.0)
            rows.append(
                {
                    "metric_type": "category_distribution_change",
                    "feature": column,
                    "category": category,
                    "reference_value": reference_pct,
                    "current_value": current_pct,
                    "absolute_change": current_pct - reference_pct,
                    "percent_change": "",
                    "psi_score": psi_score,
                    "drift_severity": drift_severity(psi_score),
                }
            )

    return pd.DataFrame(rows)


def append_prediction_shift_rows(report: pd.DataFrame, predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if "risk_probability" in predictions_df.columns:
        rows.append(
            {
                "metric_type": "prediction_distribution_shift",
                "feature": "risk_probability",
                "category": "mean",
                "reference_value": "",
                "current_value": predictions_df["risk_probability"].mean(),
                "absolute_change": "",
                "percent_change": "",
                "psi_score": "",
                "drift_severity": "low",
                "notes": "Prediction-only summary. Review trend over time when historical scoring batches exist.",
            }
        )
    if "risk_category" in predictions_df.columns:
        category_dist = predictions_df["risk_category"].value_counts(normalize=True)
        for category in ["low", "medium", "high"]:
            rows.append(
                {
                    "metric_type": "prediction_category_share",
                    "feature": "risk_category",
                    "category": category,
                    "reference_value": "",
                    "current_value": category_dist.get(category, 0.0),
                    "absolute_change": "",
                    "percent_change": "",
                    "psi_score": "",
                    "drift_severity": "low",
                    "notes": "Prediction category share for the current scoring batch.",
                }
            )
    if rows:
        return pd.concat([report, pd.DataFrame(rows)], ignore_index=True)
    return report


def save_prediction_distribution(predictions_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = ensure_parent(output_path)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "risk_probability" in predictions_df.columns:
        sns.histplot(predictions_df["risk_probability"], bins=25, ax=axes[0], color="#356d9a")
        axes[0].set_title("Predicted Risk Probability")
        axes[0].set_xlabel("Risk probability")
    else:
        axes[0].axis("off")

    if "risk_category" in predictions_df.columns:
        order = ["low", "medium", "high"]
        counts = predictions_df["risk_category"].value_counts().reindex(order, fill_value=0)
        sns.barplot(x=counts.index, y=counts.values, ax=axes[1], palette=["#4f8f75", "#d4a63f", "#b65746"], hue=counts.index, legend=False)
        axes[1].set_title("Risk Category Counts")
        axes[1].set_xlabel("Risk category")
        axes[1].set_ylabel("Accounts")
    else:
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def run_monitoring(
    reference_path: str | Path,
    current_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path = "outputs/monitoring/drift_report.csv",
) -> pd.DataFrame:
    reference_file = validate_file_exists(project_path(reference_path), "Reference data")
    current_file = validate_file_exists(project_path(current_path), "Current inference data")
    predictions_file = validate_file_exists(project_path(predictions_path), "Prediction output")
    reference_df = pd.read_csv(reference_file)
    current_df = pd.read_csv(current_file)
    predictions_df = pd.read_csv(predictions_file)
    validate_training_data(reference_df, context="Reference data")
    validate_inference_data(current_df, context="Current inference data")
    validate_prediction_output_schema(predictions_df)

    report = build_feature_drift_report(reference_df, current_df)
    report = append_prediction_shift_rows(report, predictions_df)

    output_file = ensure_parent(output_path)
    output_dir = output_file.parent
    report.to_csv(output_file, index=False)
    save_prediction_distribution(predictions_df, output_dir / "prediction_distribution.png")

    summary_path = output_dir / "monitoring_summary.txt"
    worst_scores = pd.to_numeric(report["psi_score"], errors="coerce").dropna()
    max_psi = worst_scores.max() if not worst_scores.empty else 0.0
    status = overall_drift_status(max_psi)
    high_features = report.loc[report["drift_severity"] == "high", "feature"].dropna().unique().tolist()
    moderate_features = report.loc[report["drift_severity"] == "moderate", "feature"].dropna().unique().tolist()
    title = "Customer Risk Predictor Monitoring Summary"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"{'=' * len(title)}\n")
        f.write(f"Reference rows: {len(reference_df)}\n")
        f.write(f"Current rows: {len(current_df)}\n")
        f.write(f"Maximum PSI-style drift score: {max_psi:.4f}\n")
        f.write(f"Overall data drift status: {status}\n\n")
        f.write("Severity guide: low < 0.10 PSI, moderate 0.10-0.24 PSI, high >= 0.25 PSI.\n")
        f.write("High-drift features: " + (", ".join(high_features) if high_features else "none") + "\n")
        f.write("Moderate-drift features: " + (", ".join(moderate_features) if moderate_features else "none") + "\n\n")
        f.write("Business definitions:\n")
        f.write("- Data drift: customer/account feature distributions changed compared with training data.\n")
        f.write("- Model quality drift: precision, recall, F1, or ROC-AUC changes once actual outcomes arrive.\n")
        f.write("- Concept drift: the relationship between customer behavior and churn/escalation risk changes.\n")
        f.write("\nThis report is an early-warning screen. It does not replace labeled performance monitoring.\n")

    LOGGER.info("Saved drift report to %s", output_file)
    LOGGER.info("Saved prediction distribution figure and summary under %s", ensure_dir("outputs/monitoring"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare training reference data against current inference data.")
    parser.add_argument("--reference", default="data/processed/train.csv")
    parser.add_argument("--current", default="data/inference/weekly_scoring_batch.csv")
    parser.add_argument("--predictions", default="outputs/predictions/customer_risk_predictions.csv")
    parser.add_argument("--output", default="outputs/monitoring/drift_report.csv")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_monitoring(args.reference, args.current, args.predictions, args.output)


if __name__ == "__main__":
    main()



