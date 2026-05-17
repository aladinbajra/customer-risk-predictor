from __future__ import annotations

import pandas as pd


def make_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"TEST-{i:03d}" for i in range(8)],
            "account_age_months": [4, 18, 36, 8, 60, 24, 12, 48],
            "monthly_spend": [900, 3200, 9800, 1100, 22000, 4200, 1500, 12500],
            "contract_type": ["monthly", "annual", "multi_year", "monthly", "multi_year", "annual", "monthly", "annual"],
            "region": ["Europe", "North America", "APAC", "Europe", "North America", "APAC", "Latin America", "Europe"],
            "customer_segment": ["SMB", "Mid-Market", "Enterprise", "SMB", "Strategic", "Mid-Market", "SMB", "Enterprise"],
            "product_usage_score": [35, 70, 82, 42, 91, 66, 30, 74],
            "support_ticket_count": [9, 3, 2, 8, 1, 4, 11, 3],
            "avg_response_time_hours": [12.0, 3.0, 2.0, 10.0, 1.5, 4.0, 16.0, 2.5],
            "avg_resolution_time_hours": [48.0, 12.0, 8.0, 40.0, 6.0, 16.0, 70.0, 10.0],
            "satisfaction_score": [2.1, 4.3, 4.6, 2.6, 4.8, 4.0, 1.9, 4.5],
            "failed_interactions": [4, 1, 0, 3, 0, 1, 5, 0],
            "previous_escalations": [2, 0, 0, 1, 0, 0, 3, 0],
            "renewal_due_days": [20, 140, 220, -5, 300, 60, 10, 180],
            "renewal_status": ["due_soon", "not_due", "not_due", "overdue", "not_due", "upcoming", "due_soon", "not_due"],
            "severity_low_count": [2, 2, 1, 3, 1, 2, 2, 2],
            "severity_medium_count": [3, 1, 1, 3, 0, 2, 4, 1],
            "severity_high_count": [3, 0, 0, 2, 0, 0, 4, 0],
            "severity_critical_count": [1, 0, 0, 0, 0, 0, 1, 0],
            "risk_label": [1, 0, 0, 1, 0, 0, 1, 0],
        }
    )




