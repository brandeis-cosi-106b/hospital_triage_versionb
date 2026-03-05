"""Tests for Ward and EmergencyDepartment."""

import pytest
from er_triage.ward import Ward, WardConfig, EmergencyDepartment
from er_triage.registry import PatientNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dept(**kwargs) -> EmergencyDepartment:
    return EmergencyDepartment("Test ER", **kwargs)


def _admit_batch(dept: EmergencyDepartment, specs):
    return [dept.admit(name, acuity, cc) for name, acuity, cc in specs]


# Acuities: 2, 3, 5, 6, 9, 10
_DEPT_SPECS = [
    ("Alice",   2, "sore throat"),
    ("Bob",     3, "ankle sprain"),
    ("Carol",   5, "palpitations"),
    ("Dave",    6, "abdominal pain"),
    ("Eve",     9, "chest pain — crushing"),
    ("Frank",  10, "cardiac arrest"),
]


# ---------------------------------------------------------------------------
# Ward
# ---------------------------------------------------------------------------

class TestWard:
    def _make_ward(self, capacity=20) -> Ward:
        return Ward(WardConfig(name="Test Ward", capacity=capacity))

    def test_initial_empty(self):
        w = self._make_ward()
        assert w.occupancy() == 0

    def test_admit_increments(self):
        w = self._make_ward()
        w.admit("Alice", 4, "pain")
        assert w.occupancy() == 1

    def test_discharge_decrements(self):
        w = self._make_ward()
        p = w.admit("Bob", 6, "nausea")
        w.discharge(p.patient_id)
        assert w.occupancy() == 0

    def test_capacity_enforced(self):
        w = self._make_ward(capacity=2)
        w.admit("A", 3, "c")
        w.admit("B", 4, "c")
        with pytest.raises(RuntimeError):
            w.admit("C", 5, "c")

    def test_occupancy_pct(self):
        w = self._make_ward(capacity=10)
        w.admit("A", 5, "c")
        w.admit("B", 5, "c")
        assert abs(w.occupancy_pct() - 20.0) < 0.01

    def test_is_occupied(self):
        w = self._make_ward()
        p = w.admit("C", 7, "trauma")
        assert w.is_occupied(p.patient_id)
        w.discharge(p.patient_id)
        assert not w.is_occupied(p.patient_id)

    def test_most_acute(self):
        w = self._make_ward()
        w.admit("Low",  2, "c")
        w.admit("Mid",  5, "c")
        p_high = w.admit("High", 9, "c")
        assert w.most_acute() == p_high

    def test_count_in_range(self):
        w = self._make_ward()
        w.admit("A", 2, "c")
        w.admit("B", 5, "c")
        w.admit("C", 9, "c")
        # Range [4, 6] contains only B(5)
        assert w.count_in_range(4, 6) == 1

    def test_all_patients_sorted(self):
        w = self._make_ward()
        w.admit("A", 5, "c")
        w.admit("B", 2, "c")
        w.admit("C", 8, "c")
        acuities = [p.acuity for p in w.all_patients()]
        assert acuities == sorted(acuities)


# ---------------------------------------------------------------------------
# EmergencyDepartment
# ---------------------------------------------------------------------------

class TestEmergencyDepartment:
    def test_initial_state(self):
        dept = _make_dept()
        assert dept.total_occupancy() == 0
        assert dept.general.occupancy() == 0
        assert dept.icu.occupancy() == 0

    def test_admit_goes_to_general(self):
        dept = _make_dept()
        p = dept.admit("Alice", 5, "headache")
        assert dept.general.is_occupied(p.patient_id)
        assert not dept.icu.is_occupied(p.patient_id)

    def test_discharge_from_general(self):
        dept = _make_dept()
        p = dept.admit("Bob", 3, "bruise")
        dept.discharge(p.patient_id)
        assert not dept.general.is_occupied(p.patient_id)
        assert dept.total_occupancy() == 0

    def test_discharge_unknown_raises(self):
        dept = _make_dept()
        with pytest.raises(PatientNotFoundError):
            dept.discharge("GHOST")

    def test_total_occupancy(self):
        dept = _make_dept()
        _admit_batch(dept, _DEPT_SPECS)
        assert dept.total_occupancy() == len(_DEPT_SPECS)

    def test_escalate_critical_moves_patients(self):
        dept = _make_dept()
        patients = _admit_batch(dept, _DEPT_SPECS)
        # Threshold=7: patients with acuity >= 7 → Eve(9), Frank(10)
        n = dept.escalate_critical(threshold=7)
        assert n == 2
        assert dept.icu.occupancy() == 2
        assert dept.general.occupancy() == len(_DEPT_SPECS) - 2

    def test_escalate_critical_correct_patients_in_icu(self):
        dept = _make_dept()
        _admit_batch(dept, _DEPT_SPECS)
        dept.escalate_critical(threshold=7)
        for p in dept.icu.all_patients():
            assert p.acuity >= 7

    def test_escalate_critical_general_patients_below_threshold(self):
        dept = _make_dept()
        _admit_batch(dept, _DEPT_SPECS)
        dept.escalate_critical(threshold=7)
        for p in dept.general.all_patients():
            assert p.acuity < 7

    def test_escalate_returns_count(self):
        dept = _make_dept()
        _admit_batch(dept, _DEPT_SPECS)
        n = dept.escalate_critical(threshold=7)
        assert isinstance(n, int)
        assert n >= 0

    def test_escalate_noop_when_none_qualify(self):
        dept = _make_dept()
        dept.admit("Low", 2, "c")
        dept.admit("Mid", 4, "c")
        n = dept.escalate_critical(threshold=9)
        assert n == 0
        assert dept.icu.occupancy() == 0

    def test_locate_in_general(self):
        dept = _make_dept()
        p = dept.admit("Carol", 5, "c")
        loc = dept.locate(p.patient_id)
        assert "General" in loc

    def test_locate_in_icu(self):
        dept = _make_dept()
        p = dept.admit("Dave", 9, "c")
        dept.escalate_critical(threshold=7)
        loc = dept.locate(p.patient_id)
        assert "ICU" in loc

    def test_locate_missing(self):
        dept = _make_dept()
        assert dept.locate("NOBODY") is None

    def test_all_patients_spans_both_wards(self):
        dept = _make_dept()
        admitted = _admit_batch(dept, _DEPT_SPECS)
        dept.escalate_critical(threshold=7)
        all_p = dept.all_patients()
        assert len(all_p) == len(_DEPT_SPECS)
        admitted_ids = {p.patient_id for p in admitted}
        all_ids = {p.patient_id for p in all_p}
        assert admitted_ids == all_ids

    def test_census_string(self):
        dept = _make_dept()
        _admit_batch(dept, _DEPT_SPECS)
        s = dept.census()
        assert "General" in s
        assert "ICU" in s

    def test_discharge_from_icu(self):
        dept = _make_dept()
        patients = _admit_batch(dept, _DEPT_SPECS)
        dept.escalate_critical(threshold=7)
        icu_patient = next(p for p in dept.icu.all_patients())
        dept.discharge(icu_patient.patient_id)
        assert not dept.icu.is_occupied(icu_patient.patient_id)
