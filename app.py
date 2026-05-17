from __future__ import annotations

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.score_accounts import ACTION_MAP, DEFAULT_HIGH_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD, assign_risk_category
from src.risk_metrics import predict_probabilities
from src.customer_features import ID_COLUMN, TARGET_COLUMN
from src.project_io import load_json
from src.customer_schema import REQUIRED_MODEL_INPUT_COLUMNS, validate_inference_data, validate_thresholds


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "model_registry" / "production" / "customer_risk_pipeline.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics" / "champion_metrics.json"
DRIFT_REPORT_PATH = ROOT / "outputs" / "monitoring" / "drift_report.csv"


st.set_page_config(page_title="Customer Risk Predictor", layout="wide")
st.title("Customer Risk Predictor")
st.caption("Batch scoring demo for customer success, support operations, and retention prioritization.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def make_predictions(
    model,
    df: pd.DataFrame,
    medium_threshold: float = DEFAULT_MEDIUM_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> pd.DataFrame:
    validate_thresholds(medium_threshold, high_threshold)
    scoring_df = df.drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    validate_inference_data(scoring_df, context="Uploaded scoring data")
    if ID_COLUMN not in scoring_df.columns:
        scoring_df.insert(0, ID_COLUMN, [f"UPLOAD-{i:06d}" for i in range(1, len(scoring_df) + 1)])
    probabilities = predict_probabilities(model, scoring_df)
    output = scoring_df.copy()
    output["risk_probability"] = probabilities.round(6)
    output["risk_category"] = output["risk_probability"].apply(
        assign_risk_category,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    output["recommended_action"] = output["risk_category"].map(ACTION_MAP)
    lead_columns = [ID_COLUMN, "risk_probability", "risk_category", "recommended_action"]
    return output[lead_columns + [column for column in output.columns if column not in lead_columns]]


if not MODEL_PATH.exists():
    st.error("Production model was not found. Run training first: `python src/train_risk_model.py --data data/raw/customer_risk_snapshot.csv`.")
    st.stop()

model = load_model()

with st.expander("Project overview", expanded=True):
    st.write(
        "This dashboard scores customer/account records for churn or escalation risk using the same "
        "production pipeline trained with sklearn and tracked with MLflow. The output is intended for "
        "triage: high-risk accounts should be reviewed by customer success or support leadership."
    )
    st.write(
        "Required scoring columns: "
        + ", ".join(REQUIRED_MODEL_INPUT_COLUMNS)
        + ". `customer_id` is optional and will be generated if missing."
    )

metrics_col, drift_col = st.columns([1, 1])
with metrics_col:
    st.subheader("Model Metrics")
    if METRICS_PATH.exists():
        metrics = load_json(METRICS_PATH)
        metric_cols = st.columns(5)
        for col, metric_name in zip(metric_cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            col.metric(metric_name.upper(), f"{metrics.get(metric_name, 0):.3f}")
        st.caption(f"Best model: {metrics.get('best_model', 'unknown')}")
    else:
        st.info("Metrics will appear after training.")

with drift_col:
    st.subheader("Monitoring Summary")
    if DRIFT_REPORT_PATH.exists():
        drift_df = pd.read_csv(DRIFT_REPORT_PATH)
        psi_values = pd.to_numeric(drift_df["psi_score"], errors="coerce").dropna()
        max_psi = psi_values.max() if not psi_values.empty else 0.0
        severity_col = "drift_severity" if "drift_severity" in drift_df.columns else "drift_flag"
        high_count = (drift_df[severity_col] == "high").sum()
        drift_cols = st.columns(2)
        drift_cols[0].metric("Max PSI", f"{max_psi:.3f}")
        drift_cols[1].metric("High Drift Flags", int(high_count))
        st.dataframe(drift_df.head(10), width="stretch", hide_index=True)
    else:
        st.info("Monitoring summary will appear after running `python src/customer_drift.py`.")

st.subheader("Batch Scoring")
control_cols = st.columns(2)
medium_threshold = control_cols[0].slider(
    "Medium-risk threshold",
    min_value=0.05,
    max_value=0.90,
    value=DEFAULT_MEDIUM_THRESHOLD,
    step=0.05,
)
high_threshold = control_cols[1].slider(
    "High-risk threshold",
    min_value=0.10,
    max_value=0.95,
    value=DEFAULT_HIGH_THRESHOLD,
    step=0.05,
)
uploaded_file = st.file_uploader("Upload new customer/account CSV", type=["csv"])
default_file = ROOT / "data" / "inference" / "weekly_scoring_batch.csv"

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
elif default_file.exists():
    input_df = pd.read_csv(default_file)
    st.caption(f"Using demo scoring file: {default_file.relative_to(ROOT)}")
else:
    input_df = pd.DataFrame()

if input_df.empty:
    st.warning("No scoring data available.")
    st.stop()

try:
    predictions = make_predictions(model, input_df, medium_threshold, high_threshold)
except Exception as exc:
    st.error(f"Scoring failed: {exc}")
    st.stop()

summary_cols = st.columns(3)
summary_cols[0].metric("Accounts Scored", len(predictions))
summary_cols[1].metric("High Risk", int((predictions["risk_category"] == "high").sum()))
summary_cols[2].metric("Average Risk Probability", f"{predictions['risk_probability'].mean():.3f}")

chart_df = predictions["risk_category"].value_counts().rename_axis("risk_category").reset_index(name="accounts")
chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("risk_category:N", sort=["low", "medium", "high"], title="Risk category"),
        y=alt.Y("accounts:Q", title="Accounts"),
        color=alt.Color("risk_category:N", scale=alt.Scale(domain=["low", "medium", "high"], range=["#4f8f75", "#d4a63f", "#b65746"]), legend=None),
    )
    .properties(height=280)
)
st.altair_chart(chart, width="stretch")

st.subheader("Top High-Risk Customers")
top_high_risk = predictions.sort_values("risk_probability", ascending=False).head(25)
st.dataframe(top_high_risk, width="stretch", hide_index=True)

st.subheader("Recommended Actions")
action_summary = (
    predictions.groupby(["risk_category", "recommended_action"], as_index=False)
    .size()
    .rename(columns={"size": "accounts"})
    .sort_values("accounts", ascending=False)
)
st.dataframe(action_summary, width="stretch", hide_index=True)

st.subheader("All Predictions")
st.dataframe(predictions, width="stretch", hide_index=True)



