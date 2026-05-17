# Project Report: Customer Risk Predictor

## Introduction

This project builds a local, reproducible ML/MLOps pipeline for predicting customer churn or support escalation risk. It is intentionally realistic without using private company data or claiming a live Databricks deployment.

## Business Value

Customer success and support teams often need to decide which accounts deserve immediate attention. A risk model can help prioritize accounts with low satisfaction, high ticket volume, slow resolution, failed interactions, past escalations, or upcoming renewals.

The model output is a probability plus a recommended action:

- low: monitor normally
- medium: proactive follow-up
- high: urgent retention/escalation review

Recall is especially important because missing a truly high-risk account can lead to preventable churn or a late escalation. Precision is still reported because unnecessary outreach consumes team capacity.

## Dataset

The dataset is created with `src/customer_data_factory.py` and saved to `data/raw/customer_risk_snapshot.csv`.

Local run:

- 15,000 labeled rows
- 20.0% high-risk rate
- 750 demo inference records

Features include account age, spend, contract type, region, customer segment, product usage, ticket volume, response/resolution time, satisfaction, failed interactions, previous escalations, renewal timing, and severity history.

The labels are synthetic but signal-driven. Metrics in this report come from the actual local training run.

## Feature Engineering and Preprocessing

Feature engineering adds account-level signals such as:

- tickets per account month
- spend per usage point
- escalation rate
- high-severity ticket share
- renewal window flag
- response-to-resolution gap
- support pressure score
- poor experience flag

Preprocessing uses a sklearn `ColumnTransformer`:

- median imputation for numeric features
- scaling for numeric features
- most-frequent imputation for categorical features
- one-hot encoding with unknown category handling

The train/validation/test split happens before fitting the pipeline. The pipeline is fitted only on training data, then reused for validation, test, batch inference, and the dashboard. This avoids data leakage.

## Models Compared

The project compares:

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

All models use the same feature engineering and preprocessing pipeline. Each candidate model tunes its classification threshold on the validation split, then the selected model is chosen by validation F1 with ROC-AUC and recall as tie-breakers.

## Results

| Model | Validation F1 | Validation ROC-AUC | Threshold | Test ROC-AUC | Test Precision | Test Recall | Test F1 | Test Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.884848 | 0.988034 | 0.85 | 0.991934 | 0.924188 | 0.853333 | 0.887348 | 0.956667 |
| HistGradientBoosting | 0.880811 | 0.987047 | 0.49 | 0.991609 | 0.917688 | 0.873333 | 0.894962 | 0.959000 |
| Random Forest | 0.852204 | 0.981895 | 0.58 | 0.987096 | 0.895575 | 0.843333 | 0.868670 | 0.949000 |

Selected model: `logistic_regression`

The logistic model was selected because it had the strongest validation F1 after threshold tuning. HistGradientBoosting had slightly stronger test F1 in this run, but model selection is intentionally based on validation data only to avoid tuning decisions on the holdout set.

## MLOps Design

The project includes the pieces needed to make the model reproducible and reviewable:

- reproducible data factory
- leakage-safe sklearn pipeline
- MLflow experiment tracking
- model comparison and artifact logging
- registry-style staging, production, and archive folders
- promotion metadata
- batch inference with configurable thresholds
- drift monitoring
- Streamlit dashboard
- pytest coverage for core behavior

The local registry simulates how a promoted model would move through Databricks Model Registry or Unity Catalog aliases.

## MLflow Tracking and Model Promotion

Training runs log model parameters, validation metrics, test metrics, figures, model artifacts, signatures, and input examples to local MLflow. The best model is selected using validation F1 after validation-only threshold tuning. The chosen pipeline is copied to `outputs/model_registry/production/customer_risk_pipeline.joblib`, and promotion details are written to `outputs/model_registry/production/champion_metadata.json`.

This is a local registry-style workflow. It does not create a real Unity Catalog model or production alias.

## Batch Inference and Dashboard

Batch inference loads the promoted local production pipeline, scores `data/inference/weekly_scoring_batch.csv`, assigns low/medium/high risk categories, and writes `outputs/predictions/customer_risk_predictions.csv`.

The Streamlit app provides a business-facing demo. It supports CSV upload, uses the demo scoring file by default, shows risk distribution, lists the highest-risk accounts, displays recommended actions, and surfaces saved model and monitoring metrics.

## Monitoring

Monitoring compares training reference data with current inference data and writes:

- numeric mean changes
- categorical distribution changes
- PSI-style drift scores
- prediction probability distribution
- risk category distribution

Latest local monitoring summary:

- Reference rows: 9,000
- Current rows: 750
- Maximum PSI-style drift score: 0.0652
- Overall data drift status: low

Limitations: this is feature and prediction drift monitoring. True model quality drift requires actual future labels. Concept drift can only be confirmed when the relationship between features and outcomes changes over time. Because the dataset is synthetic and signal-driven, the strong metrics should be read as evidence that the local pipeline works, not as a public churn benchmark.

## Databricks Alignment

This project is Databricks-ready, not Databricks-deployed:

- raw and processed files would become Delta tables in a real workspace
- feature logic would run as a notebook, Python wheel, or Workflow task
- MLflow would use the Databricks workspace experiment store
- the local registry would map to Unity Catalog registered models and aliases
- batch inference would run as a scheduled Databricks Workflow
- monitoring reports would be written to Delta and reviewed in dashboards
- Model Serving is future work only, and only if real-time scoring is required
- online tables are future work only, and only if real-time feature lookup is required

No real Unity Catalog permissions, Model Serving endpoints, or company workspace access are required for the local project.

## Limitations and Future Work

The dataset is synthetic and signal-driven, so the strong metrics should be read as a validation of the local pipeline, not as evidence of real-world churn performance. Real deployment would require historical churn, renewal, escalation, and support outcomes. Model quality drift also requires delayed labels; the current monitoring covers feature and prediction distribution drift only.

Useful next steps would be CI, cost-based threshold tuning, calibration checks, fairness review, scheduled retraining, and a real Databricks implementation once infrastructure and governed data are available.

## Conclusion

This project demonstrates practical ML engineering: not just model training, but the surrounding workflow needed to make a model reproducible, inspectable, and useful to a business user. The main future improvements are replacing synthetic data with real historical labels, tuning thresholds with cost analysis, adding CI, and deploying the workflow with real Databricks infrastructure.




