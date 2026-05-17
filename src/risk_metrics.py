from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.customer_features import ID_COLUMN, TARGET_COLUMN
    from src.project_io import clean_metric_dict, ensure_dir, ensure_parent, load_json, project_path, save_json, setup_logging
    from src.customer_schema import validate_file_exists, validate_training_data
except ModuleNotFoundError:
    from customer_features import ID_COLUMN, TARGET_COLUMN
    from project_io import clean_metric_dict, ensure_dir, ensure_parent, load_json, project_path, save_json, setup_logging
    from customer_schema import validate_file_exists, validate_training_data


LOGGER = logging.getLogger(__name__)


def evaluate_binary_classifier(y_true: pd.Series, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "threshold": threshold,
    }
    return clean_metric_dict(metrics)


def find_best_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 161)

    best_threshold = 0.5
    best_metrics = evaluate_binary_classifier(y_true, probabilities, threshold=best_threshold)
    for threshold in thresholds:
        metrics = evaluate_binary_classifier(y_true, probabilities, threshold=float(threshold))
        is_better = (
            metrics["f1"],
            metrics["roc_auc"],
            metrics["recall"],
            metrics["precision"],
        ) > (
            best_metrics["f1"],
            best_metrics["roc_auc"],
            best_metrics["recall"],
            best_metrics["precision"],
        )
        if is_better:
            best_threshold = float(threshold)
            best_metrics = metrics

    return round(best_threshold, 3), best_metrics


def predict_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X).astype(float)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in evaluation data.")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def save_confusion_matrix_figure(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_path: str | Path,
    threshold: float = 0.5,
    title: str = "Confusion Matrix",
) -> Path:
    output_path = ensure_parent(output_path)
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low risk", "High risk"])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_roc_curve_figure(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_path: str | Path,
    title: str = "ROC Curve",
) -> Path:
    output_path = ensure_parent(output_path)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def create_eda_artifacts(df: pd.DataFrame, figures_dir: str | Path = "outputs/figures") -> dict[str, str]:
    validate_training_data(df, context="EDA data")
    figures_dir = ensure_dir(figures_dir)
    sns.set_theme(style="whitegrid")
    artifacts: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(6, 4))
    target_counts = df[TARGET_COLUMN].value_counts(normalize=True).sort_index()
    sns.barplot(x=target_counts.index.map({0: "Low risk", 1: "High risk"}), y=target_counts.values, ax=ax)
    ax.set_ylabel("Share of accounts")
    ax.set_xlabel("")
    ax.set_title("Target Distribution")
    ax.bar_label(ax.containers[0], labels=[f"{value:.1%}" for value in target_counts.values])
    fig.tight_layout()
    path = figures_dir / "target_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts["target_distribution"] = str(path)

    numeric_columns = [
        "account_age_months",
        "monthly_spend",
        "product_usage_score",
        "support_ticket_count",
        "avg_resolution_time_hours",
        "satisfaction_score",
        "failed_interactions",
        "previous_escalations",
    ]
    numeric_columns = [column for column in numeric_columns if column in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    for ax, column in zip(axes, numeric_columns):
        sns.histplot(df[column], bins=30, kde=False, ax=ax, color="#356d9a")
        ax.set_title(column.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("")
    for ax in axes[len(numeric_columns):]:
        ax.axis("off")
    fig.suptitle("Numeric Feature Distributions", y=1.02)
    fig.tight_layout()
    path = figures_dir / "numeric_feature_distributions.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    artifacts["numeric_feature_distributions"] = str(path)

    segment_rate = df.groupby("customer_segment", dropna=False)[TARGET_COLUMN].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=segment_rate.values, y=segment_rate.index.astype(str), ax=ax, color="#b65746")
    ax.set_xlabel("High-risk rate")
    ax.set_ylabel("Customer segment")
    ax.set_title("Risk Rate by Segment")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    path = figures_dir / "risk_rate_by_segment.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts["risk_rate_by_segment"] = str(path)

    corr_df = df.select_dtypes(include=[np.number]).drop(columns=[ID_COLUMN], errors="ignore")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr_df.corr(numeric_only=True), cmap="vlag", center=0, linewidths=0.4, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    path = figures_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts["correlation_heatmap"] = str(path)

    correlations = corr_df.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(key=abs, ascending=False)
    top_correlations = correlations.head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#b65746" if value > 0 else "#356d9a" for value in top_correlations.values]
    ax.barh(top_correlations.index, top_correlations.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlation with risk_label")
    ax.set_title("Top Numeric Risk Indicators")
    fig.tight_layout()
    path = figures_dir / "top_risk_indicators.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    artifacts["top_risk_indicators"] = str(path)

    correlations.reset_index().rename(columns={"index": "feature", TARGET_COLUMN: "correlation"}).to_csv(
        figures_dir / "top_risk_indicators.csv", index=False
    )

    return artifacts


def evaluate_model_file(
    model_path: str | Path,
    data_path: str | Path,
    output_json: str | Path = "outputs/metrics/holdout_metrics.json",
    figures_dir: str | Path = "outputs/figures",
    threshold: float | None = None,
) -> dict[str, float]:
    model_file = validate_file_exists(project_path(model_path), "Model artifact")
    data_file = validate_file_exists(project_path(data_path), "Evaluation data")
    model = joblib.load(model_file)
    df = pd.read_csv(data_file)
    validate_training_data(df, context="Evaluation data")
    X, y = split_features_target(df)
    probabilities = predict_probabilities(model, X)
    if threshold is None:
        metrics_file = project_path("outputs/metrics/champion_metrics.json")
        if metrics_file.exists():
            try:
                threshold = float(load_json(metrics_file).get("threshold", 0.5))
            except Exception:
                threshold = 0.5
        else:
            threshold = 0.5
    metrics = evaluate_binary_classifier(y, probabilities, threshold=threshold)

    save_json(metrics, output_json)
    save_confusion_matrix_figure(
        y,
        probabilities,
        Path(figures_dir) / "holdout_confusion_matrix.png",
        threshold=threshold,
        title="Confusion Matrix - Evaluation",
    )
    save_roc_curve_figure(y, probabilities, Path(figures_dir) / "holdout_roc_curve.png", title="ROC Curve - Evaluation")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved customer risk model on labeled data.")
    parser.add_argument("--model", default="outputs/model_registry/production/customer_risk_pipeline.joblib")
    parser.add_argument("--data", default="data/processed/test.csv")
    parser.add_argument("--output-json", default="outputs/metrics/holdout_metrics.json")
    parser.add_argument("--figures-dir", default="outputs/figures")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    metrics = evaluate_model_file(args.model, args.data, args.output_json, args.figures_dir)
    LOGGER.info("Evaluation metrics: %s", metrics)


if __name__ == "__main__":
    main()



