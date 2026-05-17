from __future__ import annotations

import numpy as np

from src.customer_features import add_customer_risk_features

from tests.utils import make_training_frame


def test_add_customer_risk_features_creates_expected_columns():
    df = make_training_frame().drop(columns=["risk_label"])
    features = add_customer_risk_features(df)

    expected = {
        "tickets_per_account_month",
        "spend_per_usage_point",
        "escalation_rate",
        "high_severity_ticket_share",
        "renewal_window_flag",
        "response_resolution_gap",
    }
    assert expected.issubset(features.columns)
    assert np.isfinite(features["tickets_per_account_month"]).all()
    assert np.isfinite(features["high_severity_ticket_share"]).all()




