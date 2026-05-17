from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.customer_features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, FeatureEngineer
except ModuleNotFoundError:
    from customer_features import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, FeatureEngineer


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def build_feature_preprocessor_from_fitted_model(model_pipeline: Pipeline) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", model_pipeline.named_steps["features"]),
            ("preprocessor", model_pipeline.named_steps["preprocessor"]),
        ]
    )


__all__ = [
    "CATEGORICAL_FEATURES",
    "MODEL_FEATURES",
    "NUMERIC_FEATURES",
    "build_model_pipeline",
    "build_preprocessor",
    "build_feature_preprocessor_from_fitted_model",
]




