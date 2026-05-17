from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RAW_NUMERIC_FEATURES = [
    "account_age_months",
    "monthly_spend",
    "product_usage_score",
    "support_ticket_count",
    "avg_response_time_hours",
    "avg_resolution_time_hours",
    "satisfaction_score",
    "failed_interactions",
    "previous_escalations",
    "renewal_due_days",
    "severity_low_count",
    "severity_medium_count",
    "severity_high_count",
    "severity_critical_count",
]

ENGINEERED_NUMERIC_FEATURES = [
    "tickets_per_account_month",
    "spend_per_usage_point",
    "escalation_rate",
    "high_severity_ticket_share",
    "renewal_window_flag",
    "response_resolution_gap",
    "support_pressure_score",
    "poor_experience_flag",
    "renewal_support_pressure",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "region",
    "customer_segment",
    "renewal_status",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "risk_label"
ID_COLUMN = "customer_id"


def _safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def ensure_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in RAW_NUMERIC_FEATURES:
        if column not in df.columns:
            df[column] = np.nan
    for column in CATEGORICAL_FEATURES:
        if column not in df.columns:
            df[column] = "unknown"
    return df


def add_customer_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_model_columns(df)

    df["tickets_per_account_month"] = _safe_divide(
        df["support_ticket_count"], df["account_age_months"].clip(lower=1)
    )
    df["spend_per_usage_point"] = _safe_divide(
        df["monthly_spend"], df["product_usage_score"].clip(lower=1)
    )
    df["escalation_rate"] = _safe_divide(df["previous_escalations"], df["support_ticket_count"])

    severe_tickets = df["severity_high_count"].fillna(0) + df["severity_critical_count"].fillna(0)
    df["high_severity_ticket_share"] = _safe_divide(severe_tickets, df["support_ticket_count"])
    df["renewal_window_flag"] = (df["renewal_due_days"].fillna(9999) <= 90).astype(float)
    df["response_resolution_gap"] = (
        df["avg_resolution_time_hours"] - df["avg_response_time_hours"]
    )
    severe_tickets = df["severity_high_count"].fillna(0) + df["severity_critical_count"].fillna(0)
    df["support_pressure_score"] = (
        df["support_ticket_count"].fillna(0)
        + df["failed_interactions"].fillna(0) * 2.0
        + df["previous_escalations"].fillna(0) * 3.0
        + severe_tickets * 1.5
    )
    df["poor_experience_flag"] = (
        (df["satisfaction_score"].fillna(5) <= 3.6)
        & (df["product_usage_score"].fillna(100) < 55)
    ).astype(float)
    df["renewal_support_pressure"] = df["renewal_window_flag"] * (
        df["support_ticket_count"].fillna(0) + df["failed_interactions"].fillna(0)
    )

    return df.replace([np.inf, -np.inf], np.nan)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible feature transformer for customer risk signals."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        if isinstance(X, pd.DataFrame):
            self.input_columns_ = list(X.columns)
        else:
            self.input_columns_ = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            if getattr(self, "input_columns_", None):
                X = pd.DataFrame(X, columns=self.input_columns_)
            else:
                X = pd.DataFrame(X)
        return add_customer_risk_features(X)



