from __future__ import annotations

from src.customer_data_factory import generate_customer_risk_data, generate_inference_sample
from src.customer_features import TARGET_COLUMN
from src.customer_schema import REQUIRED_MODEL_INPUT_COLUMNS


def test_generate_customer_risk_data_schema_and_size():
    df = generate_customer_risk_data(rows=10000, seed=123)

    assert len(df) == 10000
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})
    for column in REQUIRED_MODEL_INPUT_COLUMNS:
        assert column in df.columns


def test_generate_inference_sample_is_unlabeled():
    df = generate_customer_risk_data(rows=10000, seed=123)
    inference_df = generate_inference_sample(df, rows=25, seed=321)

    assert len(inference_df) == 25
    assert TARGET_COLUMN not in inference_df.columns
    assert inference_df["customer_id"].str.startswith("NEW-").all()




