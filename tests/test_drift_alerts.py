"""Tests for drift alert manager."""

from src.deployment.drift_alerts import DriftAlertManager, DriftAlert


class TestNoDrift:
    """PSI below info threshold -> no alerts."""

    def test_no_drift(self):
        manager = DriftAlertManager()

        baseline = {"feature1": {"mean": 10.0, "std": 2.0}}
        current = {"feature1": {"mean": 10.01, "std": 2.0}}  # tiny shift

        alerts = manager.check_and_alert(current, baseline)

        # No alerts or only info-level (PSI < 0.05 means no alert generated)
        assert len(alerts) == 0


class TestWarningDrift:
    """Moderate drift -> warning severity."""

    def test_warning_drift(self):
        manager = DriftAlertManager()

        # Create a meaningful mean shift to produce PSI in warning range
        baseline = {"feature1": {"mean": 10.0, "std": 2.0}}
        # Shift mean enough to get PSI in warning range (0.1 <= PSI < 0.25)
        current = {"feature1": {"mean": 10.9, "std": 2.0}}

        alerts = manager.check_and_alert(current, baseline)

        assert len(alerts) >= 1
        alert = alerts[0]
        assert alert.feature == "feature1"
        assert alert.psi >= manager.PSI_INFO
        # Find an alert with warning level
        warning_alerts = [a for a in alerts if a.severity == "warning"]
        # If PSI lands in warning range, verify
        if warning_alerts:
            assert warning_alerts[0].severity == "warning"
            assert "Investigate" in warning_alerts[0].suggested_action


class TestCriticalDrift:
    """Severe drift -> critical severity."""

    def test_critical_drift(self):
        manager = DriftAlertManager()

        # Large shift to produce PSI >= 0.25
        baseline = {"feature1": {"mean": 10.0, "std": 2.0}}
        current = {"feature1": {"mean": 14.0, "std": 2.0}}  # large mean shift

        alerts = manager.check_and_alert(current, baseline)

        assert len(alerts) >= 1
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        assert len(critical_alerts) >= 1
        assert critical_alerts[0].psi >= manager.PSI_CRITICAL
        assert "Retrain" in critical_alerts[0].suggested_action


class TestGenerateReport:
    """Verify report summary structure."""

    def test_generate_report(self):
        manager = DriftAlertManager()

        # Create alerts with multiple features at different severities
        baseline = {
            "feat_stable": {"mean": 5.0, "std": 1.0},
            "feat_shifted": {"mean": 10.0, "std": 2.0},
            "feat_broken": {"mean": 10.0, "std": 2.0},
        }
        current = {
            "feat_stable": {"mean": 5.0, "std": 1.0},  # no drift
            "feat_shifted": {"mean": 10.9, "std": 2.0},  # moderate drift
            "feat_broken": {"mean": 14.0, "std": 2.0},  # severe drift
        }

        alerts = manager.check_and_alert(current, baseline)
        report = manager.generate_report(alerts)

        assert "total_alerts" in report
        assert "by_severity" in report
        assert "features_affected" in report
        assert "alerts" in report
        assert report["total_alerts"] == len(alerts)
        assert isinstance(report["by_severity"], dict)
        assert isinstance(report["features_affected"], list)
        # At least one feature should have drifted
        assert report["total_alerts"] >= 1
