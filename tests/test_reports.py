"""Tests for the reports module."""

import pytest
from er_triage import PatientRegistry, EmergencyDepartment
from er_triage.reports import (
    acuity_distribution,
    category_summary,
    critical_load,
    department_report,
    format_acuity_histogram,
    format_category_summary,
    format_load_metrics,
    format_patient_table,
    ward_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*acuities: int) -> PatientRegistry:
    reg = PatientRegistry()
    for i, a in enumerate(acuities):
        reg.admit(f"Patient_{i}", a, "chief complaint")
    return reg


def _rpt_make_dept(general_acuities, icu_acuities=()) -> EmergencyDepartment:
    dept = EmergencyDepartment("Report Test ER")
    for i, a in enumerate(general_acuities):
        dept.admit(f"Gen_{i}", a, "cc")
    for i, a in enumerate(icu_acuities):
        p = dept.general.registry.admit(f"ICU_{i}", a, "cc")
        # Manually move to ICU by transferring above a threshold below a
        dept.escalate_critical(threshold=a)
    return dept


# ---------------------------------------------------------------------------
# acuity_distribution
# ---------------------------------------------------------------------------

class TestAcuityDistribution:
    def test_empty_registry(self):
        reg = PatientRegistry()
        dist = acuity_distribution(reg)
        assert all(v == 0 for v in dist.values())
        assert set(dist.keys()) == set(range(1, 11))

    def test_counts_correct(self):
        reg = _make_registry(2, 2, 5, 5, 5, 9)
        dist = acuity_distribution(reg)
        assert dist[2] == 2
        assert dist[5] == 3
        assert dist[9] == 1
        assert dist[1] == 0

    def test_total_matches_size(self):
        acuities = [1, 3, 3, 6, 6, 6, 10]
        reg = _make_registry(*acuities)
        dist = acuity_distribution(reg)
        assert sum(dist.values()) == len(acuities)

    def test_single_patient(self):
        reg = _make_registry(7)
        dist = acuity_distribution(reg)
        assert dist[7] == 1
        assert sum(dist.values()) == 1


# ---------------------------------------------------------------------------
# category_summary
# ---------------------------------------------------------------------------

class TestCategorySummary:
    def test_all_categories_present(self):
        reg = PatientRegistry()
        summary = category_summary(reg)
        assert set(summary.keys()) == {"Green", "Yellow", "Orange", "Red"}

    def test_empty_all_zero(self):
        reg = PatientRegistry()
        summary = category_summary(reg)
        assert all(v == 0 for v in summary.values())

    def test_green_patients(self):
        reg = _make_registry(1, 2, 3)
        summary = category_summary(reg)
        assert summary["Green"] == 3

    def test_mixed_categories(self):
        # 2 green (1,2), 1 yellow (5), 1 red (10)
        reg = _make_registry(1, 2, 5, 10)
        summary = category_summary(reg)
        assert summary["Green"] == 2
        assert summary["Yellow"] == 1
        assert summary["Red"] == 1

    def test_total_matches_registry_size(self):
        reg = _make_registry(2, 4, 6, 8, 10)
        summary = category_summary(reg)
        assert sum(summary.values()) == 5


# ---------------------------------------------------------------------------
# format_patient_table
# ---------------------------------------------------------------------------

class TestFormatPatientTable:
    def test_empty(self):
        result = format_patient_table([], "Empty")
        assert "none" in result.lower() or "Empty" in result

    def test_contains_patient_name(self):
        reg = _make_registry(5)
        patients = reg.all_patients()
        result = format_patient_table(patients, "Ward")
        assert patients[0].name in result

    def test_contains_title(self):
        reg = _make_registry(3)
        result = format_patient_table(reg.all_patients(), "My Ward")
        assert "My Ward" in result


# ---------------------------------------------------------------------------
# format_acuity_histogram
# ---------------------------------------------------------------------------

class TestFormatAcuityHistogram:
    def test_returns_string(self):
        reg = _make_registry(3, 5, 7)
        result = format_acuity_histogram(reg)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_all_levels(self):
        reg = _make_registry(1)
        result = format_acuity_histogram(reg)
        for level in range(1, 11):
            assert str(level) in result


# ---------------------------------------------------------------------------
# critical_load
# ---------------------------------------------------------------------------

class TestCriticalLoad:
    def test_keys_present(self):
        dept = EmergencyDepartment("Test")
        metrics = critical_load(dept)
        assert "critical_count" in metrics
        assert "general_critical" in metrics
        assert "icu_count" in metrics
        assert "icu_pct" in metrics

    def test_empty_dept_all_zero(self):
        dept = EmergencyDepartment("Empty")
        metrics = critical_load(dept, threshold=8)
        assert metrics["critical_count"] == 0
        assert metrics["general_critical"] == 0
        assert metrics["icu_count"] == 0

    def test_icu_pct_within_range(self):
        dept = EmergencyDepartment("Test", icu_capacity=10)
        dept.admit("P1", 9, "cc")
        dept.admit("P2", 10, "cc")
        dept.escalate_critical(threshold=9)
        metrics = critical_load(dept, threshold=9)
        assert 0.0 <= metrics["icu_pct"] <= 100.0

    def test_general_only_no_icu_occupancy(self):
        dept = EmergencyDepartment("Test")
        # Acuities 2, 4, 6 — none reach threshold=8
        for a in (2, 4, 6):
            dept.admit(f"P{a}", a, "cc")
        metrics = critical_load(dept, threshold=8)
        # No one in ICU
        assert metrics["icu_count"] == 0

    def test_load_after_escalation(self):
        dept = EmergencyDepartment("Test")
        # Acuities: 2, 4, 6, 9, 10 — threshold=7, no one at 7
        for a in (2, 4, 6, 9, 10):
            dept.admit(f"P{a}", a, "cc")
        dept.escalate_critical(threshold=7)
        metrics = critical_load(dept, threshold=7)
        assert metrics["icu_count"] == 2  # 9 and 10


# ---------------------------------------------------------------------------
# Formatting smoke tests
# ---------------------------------------------------------------------------

class TestFormattingSmoke:
    def test_format_load_metrics_returns_string(self):
        dept = EmergencyDepartment("Test")
        dept.admit("X", 5, "cc")
        result = format_load_metrics(dept, threshold=8)
        assert isinstance(result, str)
        assert "8" in result

    def test_format_category_summary_returns_string(self):
        reg = _make_registry(3, 6, 9)
        result = format_category_summary(reg)
        assert isinstance(result, str)

    def test_ward_report_returns_string(self):
        dept = EmergencyDepartment("Test")
        dept.admit("A", 4, "cc")
        result = ward_report(dept.general)
        assert isinstance(result, str)

    def test_department_report_returns_string(self):
        dept = EmergencyDepartment("Test")
        dept.admit("A", 4, "cc")
        dept.admit("B", 9, "cc")
        result = department_report(dept)
        assert isinstance(result, str)
        assert "Test" in result
