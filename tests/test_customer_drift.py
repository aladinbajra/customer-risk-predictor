from __future__ import annotations

from src.customer_drift import run_monitoring

from tests.utils import make_training_frame


def test_monitoring_creates_report_and_summary(tmp_path):
    reference = make_training_frame()
    current = reference.drop(columns=["risk_label"]).copy()
    current["support_ticket_count"] = current["support_ticket_count"] + 1
    predictions = current[["customer_id"]].copy()
    predictions["risk_probability"] = [0.10, 0.20, 0.40, 0.55, 0.70, 0.80, 0.30, 0.65]
    predictions["risk_category"] = ["low", "low", "medium", "medium", "high", "high", "low", "high"]
    predictions["recommended_action"] = predictions["risk_category"].map(
        {
            "low": "monitor normally",
            "medium": "proactive follow-up",
            "high": "urgent retention/escalation review",
        }
    )

    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    predictions_path = tmp_path / "predictions.csv"
    report_path = tmp_path / "monitoring" / "drift_report.csv"
    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    report = run_monitoring(reference_path, current_path, predictions_path, report_path)

    assert report_path.exists()
    assert (report_path.parent / "monitoring_summary.txt").exists()
    assert (report_path.parent / "prediction_distribution.png").exists()
    assert "drift_severity" in report.columns
    assert set(report["drift_severity"]).issubset({"low", "moderate", "high"})




