from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.customer_features import TARGET_COLUMN
    from src.project_io import ensure_parent, project_path, setup_logging
except ModuleNotFoundError:
    from customer_features import TARGET_COLUMN
    from project_io import ensure_parent, project_path, setup_logging


LOGGER = logging.getLogger(__name__)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def generate_customer_risk_data(rows: int = 15000, seed: int = 42) -> pd.DataFrame:
    if rows < 10000:
        raise ValueError("This portfolio project expects at least 10,000 rows.")

    rng = np.random.default_rng(seed)

    segments = np.array(["SMB", "Mid-Market", "Enterprise", "Strategic"])
    contracts = np.array(["monthly", "annual", "multi_year"])
    regions = np.array(["North America", "Europe", "APAC", "Latin America", "Middle East & Africa"])

    customer_segment = rng.choice(segments, size=rows, p=[0.38, 0.31, 0.23, 0.08])
    contract_type = rng.choice(contracts, size=rows, p=[0.34, 0.46, 0.20])
    region = rng.choice(regions, size=rows, p=[0.36, 0.28, 0.20, 0.10, 0.06])

    account_age_months = np.clip(rng.gamma(shape=3.2, scale=16, size=rows).round(), 1, 120).astype(int)

    segment_spend = {
        "SMB": 850,
        "Mid-Market": 3200,
        "Enterprise": 10500,
        "Strategic": 24500,
    }
    contract_multiplier = {"monthly": 0.78, "annual": 1.0, "multi_year": 1.22}
    region_multiplier = {
        "North America": 1.12,
        "Europe": 1.03,
        "APAC": 0.92,
        "Latin America": 0.78,
        "Middle East & Africa": 0.84,
    }
    base_spend = np.array([segment_spend[x] for x in customer_segment], dtype=float)
    monthly_spend = (
        base_spend
        * np.array([contract_multiplier[x] for x in contract_type])
        * np.array([region_multiplier[x] for x in region])
        * rng.lognormal(mean=0, sigma=0.34, size=rows)
    )
    monthly_spend = np.round(monthly_spend, 2)

    usage_base = rng.normal(loc=67, scale=17, size=rows)
    usage_base += np.where(contract_type == "multi_year", 5, 0)
    usage_base += np.where(contract_type == "monthly", -5, 0)
    usage_base += np.where(account_age_months < 6, -8, 0)
    usage_base += np.where(customer_segment == "Strategic", 4, 0)
    product_usage_score = np.clip(usage_base, 1, 100).round(2)

    low_usage_pressure = (100 - product_usage_score) / 100
    support_lambda = (
        1.5
        + (monthly_spend / 7000)
        + low_usage_pressure * 7.5
        + np.where(customer_segment == "Enterprise", 1.1, 0)
        + np.where(customer_segment == "Strategic", 1.7, 0)
        + np.where(contract_type == "monthly", 0.7, 0)
    )
    support_ticket_count = np.clip(rng.poisson(support_lambda), 0, 60).astype(int)

    failed_lambda = 0.35 + low_usage_pressure * 2.3 + support_ticket_count * 0.10
    failed_interactions = np.clip(rng.poisson(failed_lambda), 0, 25).astype(int)

    avg_response_time_hours = rng.gamma(shape=2.0, scale=3.2, size=rows)
    avg_response_time_hours += support_ticket_count * 0.25 + failed_interactions * 0.75
    avg_response_time_hours += np.where(customer_segment == "Strategic", -1.8, 0)
    avg_response_time_hours += np.where(region == "Middle East & Africa", 1.6, 0)
    avg_response_time_hours = np.clip(avg_response_time_hours, 0.25, 96).round(2)

    previous_escalations_lambda = (
        0.08 + support_ticket_count * 0.035 + failed_interactions * 0.08 + low_usage_pressure * 0.55
    )
    previous_escalations = np.clip(rng.poisson(previous_escalations_lambda), 0, 12).astype(int)

    critical_probability = np.clip(0.025 + low_usage_pressure * 0.05 + failed_interactions * 0.006, 0.01, 0.28)
    severity_critical_count = rng.binomial(support_ticket_count, critical_probability)
    remaining_tickets = support_ticket_count - severity_critical_count
    high_probability = np.clip(0.10 + low_usage_pressure * 0.13 + failed_interactions * 0.010, 0.04, 0.42)
    severity_high_count = rng.binomial(remaining_tickets, high_probability)
    remaining_tickets = remaining_tickets - severity_high_count
    severity_medium_count = rng.binomial(remaining_tickets, 0.42)
    severity_low_count = remaining_tickets - severity_medium_count

    avg_resolution_time_hours = avg_response_time_hours * rng.uniform(2.0, 4.2, rows)
    avg_resolution_time_hours += severity_high_count * 4.0 + severity_critical_count * 9.0
    avg_resolution_time_hours = np.clip(avg_resolution_time_hours, 1.0, 240).round(2)

    satisfaction_score = (
        4.75
        + product_usage_score / 100 * 0.65
        - support_ticket_count * 0.045
        - failed_interactions * 0.14
        - avg_response_time_hours * 0.018
        - severity_high_count * 0.06
        - severity_critical_count * 0.13
        + rng.normal(0, 0.35, rows)
    )
    satisfaction_score = np.clip(satisfaction_score, 1, 5).round(2)

    renewal_due_days = rng.integers(-30, 366, rows)
    renewal_status = np.select(
        [
            renewal_due_days < 0,
            renewal_due_days <= 45,
            renewal_due_days <= 120,
        ],
        ["overdue", "due_soon", "upcoming"],
        default="not_due",
    )

    risk_score = (
        support_ticket_count * 0.115
        + avg_response_time_hours * 0.030
        + avg_resolution_time_hours * 0.007
        - product_usage_score * 0.030
        - satisfaction_score * 0.78
        + failed_interactions * 0.46
        + previous_escalations * 0.78
        + severity_high_count * 0.31
        + severity_critical_count * 0.72
        + np.where(contract_type == "monthly", 0.80, 0)
        + np.where(contract_type == "multi_year", -0.45, 0)
        + np.where(customer_segment == "SMB", 0.28, 0)
        + np.where(customer_segment == "Strategic", -0.18, 0)
        + np.where(renewal_due_days < 0, 1.28, 0)
        + np.where((renewal_due_days >= 0) & (renewal_due_days <= 45), 0.82, 0)
        + np.where(account_age_months < 6, 0.62, 0)
        + np.where((monthly_spend > np.quantile(monthly_spend, 0.85)) & (product_usage_score < 45), 0.90, 0)
        + np.where((support_ticket_count >= 9) & (satisfaction_score <= 3.6), 0.95, 0)
        + np.where((failed_interactions >= 4) & (previous_escalations >= 1), 0.85, 0)
        + np.where((severity_critical_count >= 2) | (severity_high_count >= 5), 0.88, 0)
    )
    risk_signal = risk_score + rng.normal(0, 0.55, rows)
    risk_cutoff = np.quantile(risk_signal, 0.80)
    risk_label = (risk_signal >= risk_cutoff).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(1, rows + 1)],
            "account_age_months": account_age_months,
            "monthly_spend": monthly_spend,
            "contract_type": contract_type,
            "region": region,
            "customer_segment": customer_segment,
            "product_usage_score": product_usage_score,
            "support_ticket_count": support_ticket_count,
            "avg_response_time_hours": avg_response_time_hours,
            "avg_resolution_time_hours": avg_resolution_time_hours,
            "satisfaction_score": satisfaction_score,
            "failed_interactions": failed_interactions,
            "previous_escalations": previous_escalations,
            "renewal_due_days": renewal_due_days,
            "renewal_status": renewal_status,
            "severity_low_count": severity_low_count,
            "severity_medium_count": severity_medium_count,
            "severity_high_count": severity_high_count,
            "severity_critical_count": severity_critical_count,
            TARGET_COLUMN: risk_label,
        }
    )

    missing_numeric = [
        "monthly_spend",
        "product_usage_score",
        "avg_response_time_hours",
        "avg_resolution_time_hours",
        "satisfaction_score",
    ]
    for column in missing_numeric:
        mask = rng.random(rows) < 0.015
        df.loc[mask, column] = np.nan
    for column in ["contract_type", "region", "renewal_status"]:
        mask = rng.random(rows) < 0.01
        df.loc[mask, column] = np.nan

    return df


def generate_inference_sample(df: pd.DataFrame, rows: int = 750, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_size = min(rows, len(df))
    sample = df.sample(sample_size, random_state=seed).copy().reset_index(drop=True)
    if TARGET_COLUMN in sample.columns:
        sample = sample.drop(columns=[TARGET_COLUMN])

    drift_mask = rng.random(sample_size) < 0.28
    sample.loc[drift_mask, "support_ticket_count"] = (
        sample.loc[drift_mask, "support_ticket_count"].fillna(0) + rng.integers(1, 5, drift_mask.sum())
    )
    sample.loc[drift_mask, "failed_interactions"] = (
        sample.loc[drift_mask, "failed_interactions"].fillna(0) + rng.integers(0, 3, drift_mask.sum())
    )
    sample.loc[drift_mask, "satisfaction_score"] = (
        sample.loc[drift_mask, "satisfaction_score"].fillna(3.5) - rng.uniform(0.1, 0.8, drift_mask.sum())
    ).clip(lower=1, upper=5)
    sample.loc[drift_mask, "product_usage_score"] = (
        sample.loc[drift_mask, "product_usage_score"].fillna(60) - rng.uniform(2, 12, drift_mask.sum())
    ).clip(lower=1, upper=100)
    sample["customer_id"] = [f"NEW-{i:06d}" for i in range(1, sample_size + 1)]
    return sample


def write_dataset(df: pd.DataFrame, output: str | Path) -> Path:
    output_path = ensure_parent(output)
    df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a realistic synthetic customer risk dataset.")
    parser.add_argument("--rows", type=int, default=15000, help="Number of labeled customer/account rows.")
    parser.add_argument("--output", default="data/raw/customer_risk_snapshot.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--inference-output",
        default="data/inference/weekly_scoring_batch.csv",
        help="Output CSV path for unlabeled batch inference sample.",
    )
    parser.add_argument("--inference-rows", type=int, default=750, help="Rows to save for batch inference.")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    df = generate_customer_risk_data(rows=args.rows, seed=args.seed)
    output_path = write_dataset(df, args.output)

    inference_df = generate_inference_sample(df, rows=args.inference_rows, seed=args.seed + 99)
    inference_path = write_dataset(inference_df, args.inference_output)

    risk_rate = df[TARGET_COLUMN].mean()
    LOGGER.info("Saved labeled dataset to %s", project_path(output_path))
    LOGGER.info("Saved inference sample to %s", project_path(inference_path))
    LOGGER.info("Rows: %s | High-risk rate: %.3f", len(df), risk_rate)


if __name__ == "__main__":
    main()



