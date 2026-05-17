from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from src.risk_metrics import split_features_target
from src.risk_pipeline import build_model_pipeline

from tests.utils import make_training_frame


def test_customer_risk_pipeline_fits_and_scores():
    df = make_training_frame()
    X, y = split_features_target(df)
    pipeline = build_model_pipeline(LogisticRegression(max_iter=500))

    pipeline.fit(X, y)
    probabilities = pipeline.predict_proba(X)[:, 1]

    assert len(probabilities) == len(df)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert "preprocessor" in pipeline.named_steps




