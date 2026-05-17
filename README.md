# Customer Risk Predictor

This project predicts which customer accounts are at high risk of churn or support escalation, then turns the prediction into an action a customer success or support team can use. It is designed as a practical MLOps portfolio project: clean local execution, MLflow tracking, batch inference, drift monitoring, a Streamlit dashboard, and Databricks-ready notebooks/docs without requiring private company data or live Databricks access.

## Business Problem

Customer success and support teams need an early warning system for accounts that may churn, fail to renew, or require escalation review. The model predicts `risk_label`:

- `1`: high risk
- `0`: low risk

Recall matters in this use case because a false negative can mean a missed retention opportunity. Precision still matters because customer success teams have limited review capacity, so this project reports both.

## Architecture

```text
Synthetic account/support data
  -> data/raw/customer_risk_snapshot.csv
  -> EDA and train/validation/test split
  -> sklearn FeatureEngineer + ColumnTransformer + model
  -> MLflow model comparison
  -> local registry-style promotion
  -> batch predictions, monitoring reports, Streamlit dashboard
```

## What Makes This Project MLOps, Not Just ML?

This project includes the operational pieces around the model, not only a training script:

- reproducible synthetic data generation with a fixed random seed
- leakage-safe train/validation/test splitting before preprocessing is fitted
- a reusable sklearn pipeline for feature engineering, imputation, encoding, scaling, and modeling
- MLflow experiment tracking with model comparison, metrics, figures, signatures, and input examples
- local registry-style model promotion with staging, production, archive folders, and promotion metadata
- batch scoring that loads the promoted production pipeline
- drift monitoring for feature and prediction distribution changes
- Streamlit dashboard for business-facing review of predictions and monitoring outputs
- pytest coverage for data, features, preprocessing, inference, and monitoring behavior
- Databricks-ready notebooks and documentation without claiming a live Databricks deployment

## Project Structure

```text
customer-risk-predictor/
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- inference/
|-- notebooks/
|-- src/
|-- tests/
|-- outputs/
|   |-- figures/
|   |-- metrics/
|   |-- model_registry/
|   |-- models/
|   |-- monitoring/
|   `-- predictions/
|-- app.py
|-- README.md
`-- PROJECT_REPORT.md
```

## Setup

Windows PowerShell:

```powershell
cd customer-risk-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
cd customer-risk-predictor
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Generate Data

```bash
python src/customer_data_factory.py --rows 15000 --output data/raw/customer_risk_snapshot.csv
```

This writes:

- `data/raw/customer_risk_snapshot.csv`
- `data/inference/weekly_scoring_batch.csv`

## Train Models

```bash
python src/train_risk_model.py --data data/raw/customer_risk_snapshot.csv --experiment-name customer_risk_predictor
```

The training script:

- creates EDA figures
- creates stratified train/validation/test splits
- fits preprocessing only on the training split
- compares Logistic Regression, Random Forest, and HistGradientBoosting
- logs parameters, metrics, figures, model artifacts, signatures, and input examples to MLflow
- promotes the best validation model to the local production registry folder

## Evaluate

```bash
python src/risk_metrics.py --model outputs/model_registry/production/customer_risk_pipeline.joblib --data data/processed/test.csv
```

## MLflow

```bash
mlflow ui
```

Open the URL shown by MLflow, usually `http://127.0.0.1:5000`. The experiment name is:

```text
customer_risk_predictor
```

Note: local MLflow uses the `mlruns/` folder. That folder is ignored by git because runs are reproducible by rerunning training.

## Batch Inference

```bash
python src/score_accounts.py --input data/inference/weekly_scoring_batch.csv --output outputs/predictions/customer_risk_predictions.csv
```

Risk thresholds are configurable:

```bash
python src/score_accounts.py --input data/inference/weekly_scoring_batch.csv --output outputs/predictions/customer_risk_predictions.csv --medium-threshold 0.35 --high-threshold 0.65
```

Output columns include:

- `customer_id`
- `risk_probability`
- `risk_category`
- `recommended_action`

Recommended actions:

- low: monitor normally
- medium: proactive follow-up
- high: urgent retention/escalation review

## Monitoring

```bash
python src/customer_drift.py --reference data/processed/train.csv --current data/inference/weekly_scoring_batch.csv --predictions outputs/predictions/customer_risk_predictions.csv
```

Monitoring outputs:

- `outputs/monitoring/drift_report.csv`
- `outputs/monitoring/prediction_distribution.png`
- `outputs/monitoring/monitoring_summary.txt`

PSI-style severity labels:

- low: PSI < 0.10
- moderate: PSI 0.10 to 0.24
- high: PSI >= 0.25

## Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard supports CSV upload, uses the demo scoring file by default, scores accounts with the production pipeline, shows risk distribution, lists top high-risk accounts, summarizes recommended actions, and displays model/monitoring outputs when available.

## Tests

```bash
python -m pytest -q
```

Tests cover data factory, feature engineering, preprocessing, batch inference output schema, and monitoring report creation.

## Final Local Results

The table below comes from the latest local training run saved in `outputs/metrics/model_scorecard.csv`.

Dataset:

- Rows: 15,000
- High-risk rate: 20.0%
- Train/validation/test: 9,000 / 3,000 / 3,000

| Model | Validation F1 | Validation ROC-AUC | Threshold | Test ROC-AUC | Test Precision | Test Recall | Test F1 | Test Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.884848 | 0.988034 | 0.85 | 0.991934 | 0.924188 | 0.853333 | 0.887348 | 0.956667 |
| HistGradientBoosting | 0.880811 | 0.987047 | 0.49 | 0.991609 | 0.917688 | 0.873333 | 0.894962 | 0.959000 |
| Random Forest | 0.852204 | 0.981895 | 0.58 | 0.987096 | 0.895575 | 0.843333 | 0.868670 | 0.949000 |

Selected model: `logistic_regression`

The selected model had the best validation F1 after threshold tuning on the validation split. The reported test metrics use that validation-selected threshold; the test set is not used to choose the threshold.

## Local Model Registry Simulation

```text
outputs/model_registry/
|-- archive/
|-- staging/
`-- production/
```

The promoted production model is:

```text
outputs/model_registry/production/customer_risk_pipeline.joblib
```

Promotion metadata is saved to:

```text
outputs/model_registry/production/champion_metadata.json
```

This simulates Databricks/Unity Catalog model promotion locally. It does not claim a real Unity Catalog deployment.

## Databricks Mapping

The project is Databricks-ready, not Databricks-deployed.

- In a Databricks implementation, CSV datasets would be replaced with Delta tables.
- `src/customer_features.py` would become a notebook, Python wheel, or Workflow task.
- MLflow tracking would use the Databricks workspace experiment store.
- Local registry folders would map to Unity Catalog registered models and aliases.
- Batch inference would run as a Databricks Workflow task.
- Monitoring would run as a scheduled job that writes reports to Delta.
- Model Serving is future work only and would be added only if real-time scoring is needed.
- Online tables are future work only and are useful only if real-time feature lookup is needed.

## Validation Commands

```bash
python -m pip install -r requirements.txt
python src/customer_data_factory.py --rows 15000 --output data/raw/customer_risk_snapshot.csv
python src/train_risk_model.py --data data/raw/customer_risk_snapshot.csv --experiment-name customer_risk_predictor
python src/risk_metrics.py --model outputs/model_registry/production/customer_risk_pipeline.joblib --data data/processed/test.csv
python src/score_accounts.py --input data/inference/weekly_scoring_batch.csv --output outputs/predictions/customer_risk_predictions.csv
python src/customer_drift.py --reference data/processed/train.csv --current data/inference/weekly_scoring_batch.csv --predictions outputs/predictions/customer_risk_predictions.csv
python -m pytest -q
```

Compile check on macOS/Linux:

```bash
python -m py_compile src/*.py app.py tests/*.py
```

Compile check on PowerShell:

```powershell
Get-ChildItem src, tests -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
python -m py_compile app.py
```

## Limitations and Future Work

- The dataset is synthetic and signal-driven, so the strong metrics should not be treated as a real churn benchmark.
- Real deployment would require historical churn, renewal, escalation, and support outcomes.
- The current classification threshold is tuned on validation F1; a real deployment should tune it with business cost and team capacity.
- Model quality drift requires delayed labels; current monitoring focuses on feature and prediction distribution drift.
- Add CI, scheduled retraining, calibration, and fairness checks before production use.
- Move to Databricks Workflows, Delta tables, and Unity Catalog once real infrastructure is available.




