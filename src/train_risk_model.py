from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.risk_metrics import (
        create_eda_artifacts,
        evaluate_binary_classifier,
        find_best_threshold,
        predict_probabilities,
        save_confusion_matrix_figure,
        save_roc_curve_figure,
        split_features_target,
    )
    from src.customer_features import RAW_NUMERIC_FEATURES, TARGET_COLUMN
    from src.risk_pipeline import build_feature_preprocessor_from_fitted_model, build_model_pipeline
    from src.project_io import clean_metric_dict, ensure_dir, ensure_parent, project_path, save_json, setup_logging
    from src.customer_schema import validate_file_exists, validate_training_data
except ModuleNotFoundError:
    from risk_metrics import (
        create_eda_artifacts,
        evaluate_binary_classifier,
        find_best_threshold,
        predict_probabilities,
        save_confusion_matrix_figure,
        save_roc_curve_figure,
        split_features_target,
    )
    from customer_features import RAW_NUMERIC_FEATURES, TARGET_COLUMN
    from risk_pipeline import build_feature_preprocessor_from_fitted_model, build_model_pipeline
    from project_io import clean_metric_dict, ensure_dir, ensure_parent, project_path, save_json, setup_logging
    from customer_schema import validate_file_exists, validate_training_data


LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_training_data(df)

    train_val_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_val_df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        path = ensure_parent(f"data/processed/{name}.csv")
        split_df.to_csv(path, index=False)
        LOGGER.info("Saved %s split to %s (%s rows)", name, path, len(split_df))


def get_model_specs() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=260,
            max_depth=14,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.04,
            random_state=RANDOM_STATE,
        ),
    }


def _log_model_params(model_name: str, estimator: Any) -> None:
    params = estimator.get_params()
    safe_params = {
        f"{model_name}_{key}": value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    mlflow.log_params(safe_params)


def _signature_sample(X_sample: pd.DataFrame) -> pd.DataFrame:
    sample = X_sample.copy()
    for column in RAW_NUMERIC_FEATURES:
        if column in sample.columns:
            sample[column] = sample[column].astype(float)
    return sample


def _log_candidate_model(pipeline, X_sample: pd.DataFrame, model_name: str) -> None:
    try:
        signature_sample = _signature_sample(X_sample)
        prediction_sample = predict_probabilities(pipeline, signature_sample)
        signature = infer_signature(signature_sample, prediction_sample)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=f"{model_name}_pipeline",
            signature=signature,
            input_example=signature_sample.head(5),
        )
    except Exception as exc:
        LOGGER.warning("Could not log MLflow model signature for %s: %s", model_name, exc)
        mlflow.sklearn.log_model(sk_model=pipeline, name=f"{model_name}_pipeline")


def _log_evaluation_figures(
    model_name: str,
    y_validation: pd.Series,
    validation_probabilities,
    y_test: pd.Series,
    test_probabilities,
    threshold: float,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        save_confusion_matrix_figure(
            y_validation,
            validation_probabilities,
            temp_dir_path / f"{model_name}_validation_champion_confusion_matrix.png",
            threshold=threshold,
            title=f"Validation Confusion Matrix - {model_name}",
        )
        save_roc_curve_figure(
            y_validation,
            validation_probabilities,
            temp_dir_path / f"{model_name}_validation_champion_roc_curve.png",
            title=f"Validation ROC Curve - {model_name}",
        )
        save_confusion_matrix_figure(
            y_test,
            test_probabilities,
            temp_dir_path / f"{model_name}_test_champion_confusion_matrix.png",
            threshold=threshold,
            title=f"Test Confusion Matrix - {model_name}",
        )
        save_roc_curve_figure(
            y_test,
            test_probabilities,
            temp_dir_path / f"{model_name}_test_champion_roc_curve.png",
            title=f"Test ROC Curve - {model_name}",
        )
        mlflow.log_artifacts(str(temp_dir_path), artifact_path="figures")


def train_and_compare(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    experiment_name: str,
    tracking_uri: str,
) -> tuple[str, Any, pd.DataFrame, dict[str, float], str]:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, y_train = split_features_target(train_df)
    X_validation, y_validation = split_features_target(validation_df)
    X_test, y_test = split_features_target(test_df)

    comparison_rows = []
    trained_models: dict[str, Any] = {}
    run_ids: dict[str, str] = {}

    for model_name, estimator in get_model_specs().items():
        LOGGER.info("Training %s", model_name)
        pipeline = build_model_pipeline(estimator)
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("train_rows", len(train_df))
            mlflow.log_param("validation_rows", len(validation_df))
            mlflow.log_param("test_rows", len(test_df))
            _log_model_params(model_name, estimator)

            pipeline.fit(X_train, y_train)

            validation_probabilities = predict_probabilities(pipeline, X_validation)
            default_validation_metrics = evaluate_binary_classifier(y_validation, validation_probabilities)
            best_threshold, validation_metrics = find_best_threshold(y_validation, validation_probabilities)
            mlflow.log_metrics({f"validation_{key}": value for key, value in validation_metrics.items()})
            mlflow.log_metrics(
                {f"validation_default_{key}": value for key, value in default_validation_metrics.items()}
            )
            mlflow.log_param("selected_threshold", best_threshold)

            test_probabilities = predict_probabilities(pipeline, X_test)
            test_metrics = evaluate_binary_classifier(y_test, test_probabilities, threshold=best_threshold)
            mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
            _log_evaluation_figures(
                model_name,
                y_validation,
                validation_probabilities,
                y_test,
                test_probabilities,
                threshold=best_threshold,
            )
            _log_candidate_model(pipeline, X_train.head(20), model_name)

            comparison_rows.append(
                {
                    "model": model_name,
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                }
            )

            trained_models[model_name] = pipeline
            run_ids[model_name] = run.info.run_id

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["validation_f1", "validation_roc_auc", "validation_recall"],
        ascending=False,
    )
    best_model_name = comparison_df.iloc[0]["model"]
    best_pipeline = trained_models[best_model_name]
    best_run_id = run_ids[best_model_name]

    final_probabilities = predict_probabilities(best_pipeline, X_test)
    best_threshold = float(comparison_df.iloc[0]["validation_threshold"])
    final_metrics = evaluate_binary_classifier(y_test, final_probabilities, threshold=best_threshold)
    final_metrics = clean_metric_dict(
        {
            **final_metrics,
            "best_model": best_model_name,
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
        }
    )

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_param("promoted_to_local_production", True)
        mlflow.log_metrics({f"final_{key}": value for key, value in final_metrics.items() if isinstance(value, float)})
        try:
            signature_sample = _signature_sample(X_train.head(20))
            signature_output = predict_probabilities(best_pipeline, signature_sample)
            signature = infer_signature(signature_sample, signature_output)
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                name="champion_customer_risk_model",
                signature=signature,
                input_example=signature_sample.head(5),
            )
        except Exception as exc:
            LOGGER.warning("Could not log MLflow model signature: %s", exc)
            mlflow.sklearn.log_model(sk_model=best_pipeline, name="champion_customer_risk_model")

    return best_model_name, best_pipeline, comparison_df, final_metrics, best_run_id


def save_model_artifacts(best_pipeline, final_metrics: dict[str, Any], comparison_df: pd.DataFrame) -> None:
    metrics_path = ensure_parent("outputs/metrics/model_scorecard.csv")
    comparison_df.to_csv(metrics_path, index=False)

    final_metrics_path = save_json(final_metrics, "outputs/metrics/champion_metrics.json")

    model_path = ensure_parent("outputs/models/customer_risk_pipeline.joblib")
    joblib.dump(best_pipeline, model_path)

    feature_preprocessor = build_feature_preprocessor_from_fitted_model(best_pipeline)
    feature_preprocessor_path = ensure_parent("outputs/models/customer_feature_preprocessor.joblib")
    joblib.dump(feature_preprocessor, feature_preprocessor_path)

    archive_dir = ensure_dir("outputs/model_registry/archive")
    staging_path = ensure_parent("outputs/model_registry/staging/customer_risk_pipeline.joblib")
    production_path = ensure_parent("outputs/model_registry/production/customer_risk_pipeline.joblib")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if production_path.exists():
        archive_path = archive_dir / f"customer_risk_pipeline_{timestamp}.joblib"
        shutil.copy2(production_path, archive_path)
    shutil.copy2(model_path, staging_path)
    shutil.copy2(model_path, production_path)

    best_model_name = str(final_metrics["best_model"])
    best_row = comparison_df.loc[comparison_df["model"] == best_model_name].iloc[0].to_dict()
    estimator = best_pipeline.named_steps["model"]
    root_path = project_path(".").resolve()
    registry_metadata = {
        "model_name": best_model_name,
        "model_type": estimator.__class__.__name__,
        "promotion_date_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Highest validation F1 after validation-tuned threshold, then ROC-AUC and recall.",
        "selection_metric": "validation_f1",
        "validation_metrics": {
            key.replace("validation_", ""): clean_metric_dict({"value": value})["value"]
            for key, value in best_row.items()
            if key.startswith("validation_")
        },
        "test_metrics": {
            key.replace("test_", ""): clean_metric_dict({"value": value})["value"]
            for key, value in best_row.items()
            if key.startswith("test_")
        },
        "artifact_path": production_path.resolve().relative_to(root_path).as_posix(),
        "staging_artifact_path": staging_path.resolve().relative_to(root_path).as_posix(),
        "source_model_path": model_path.resolve().relative_to(root_path).as_posix(),
        "local_registry_note": "This folder simulates Databricks/Unity Catalog model promotion for local use.",
    }
    metadata_path = save_json(registry_metadata, "outputs/model_registry/production/champion_metadata.json")
    save_json(registry_metadata, "outputs/model_registry/staging/champion_metadata.json")

    LOGGER.info("Saved model comparison to %s", metrics_path)
    LOGGER.info("Saved final metrics to %s", final_metrics_path)
    LOGGER.info("Saved final model to %s and promoted local production copy to %s", model_path, production_path)
    LOGGER.info("Saved fitted feature preprocessor to %s", feature_preprocessor_path)
    LOGGER.info("Saved model registry metadata to %s", metadata_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Customer Risk Predictor models with MLflow tracking.")
    parser.add_argument("--data", default="data/raw/customer_risk_snapshot.csv", help="Path to labeled CSV data.")
    parser.add_argument("--experiment-name", default="customer_risk_predictor", help="MLflow experiment name.")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    data_path = validate_file_exists(project_path(args.data), "Training data")

    ensure_dir("outputs/metrics")
    ensure_dir("outputs/figures")
    ensure_dir("outputs/models")
    ensure_dir("outputs/model_registry/staging")
    ensure_dir("outputs/model_registry/production")

    df = pd.read_csv(data_path)
    validate_training_data(df)
    LOGGER.info("Loaded %s rows from %s", len(df), data_path)
    LOGGER.info("High-risk label rate: %.3f", df[TARGET_COLUMN].mean())

    create_eda_artifacts(df, "outputs/figures")

    train_df, validation_df, test_df = split_dataset(df)
    save_splits(train_df, validation_df, test_df)

    best_model_name, best_pipeline, comparison_df, final_metrics, best_run_id = train_and_compare(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
    )

    X_test, y_test = split_features_target(test_df)
    final_probabilities = predict_probabilities(best_pipeline, X_test)
    final_threshold = float(final_metrics.get("threshold", 0.5))
    confusion_path = save_confusion_matrix_figure(
        y_test,
        final_probabilities,
        "outputs/figures/champion_confusion_matrix.png",
        threshold=final_threshold,
    )
    roc_path = save_roc_curve_figure(y_test, final_probabilities, "outputs/figures/champion_roc_curve.png")

    save_model_artifacts(best_pipeline, final_metrics, comparison_df)

    with mlflow.start_run(run_id=best_run_id):
        mlflow.log_artifact(str(project_path(confusion_path)), artifact_path="figures")
        mlflow.log_artifact(str(project_path(roc_path)), artifact_path="figures")
        mlflow.log_artifact(str(project_path("outputs/metrics/model_scorecard.csv")), artifact_path="metrics")

    LOGGER.info("Best model: %s | Final metrics: %s", best_model_name, final_metrics)


if __name__ == "__main__":
    main()



