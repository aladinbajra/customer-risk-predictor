from __future__ import annotations

import joblib
from sklearn.linear_model import LogisticRegression

from src.score_accounts import run_score_accounts
from src.risk_metrics import split_features_target
from src.risk_pipeline import build_model_pipeline

from tests.utils import make_training_frame


def test_score_accounts_output_schema(tmp_path):
    train_df = make_training_frame()
    X, y = split_features_target(train_df)
    model = build_model_pipeline(LogisticRegression(max_iter=500))
    model.fit(X, y)

    model_path = tmp_path / "model.joblib"
    input_path = tmp_path / "weekly_scoring_batch.csv"
    output_path = tmp_path / "predictions.csv"
    joblib.dump(model, model_path)
    X.head(4).to_csv(input_path, index=False)

    predictions = run_score_accounts(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        medium_threshold=0.30,
        high_threshold=0.60,
    )

    assert output_path.exists()
    assert ["customer_id", "risk_probability", "risk_category", "recommended_action"] == list(predictions.columns[:4])
    assert set(predictions["risk_category"]).issubset({"low", "medium", "high"})




