"""Tests for PatientRegistry."""

import pytest
from er_triage.registry import (
    DuplicateAdmissionError,
    PatientNotFoundError,
    PatientRegistry,
)
from er_triage.patient import Patient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _populate(reg: PatientRegistry, specs: list[tuple[str, int, str]]) -> list[Patient]:
    """Admit a batch of patients and return their Patient records."""
    return [reg.admit(name, acuity, complaint) for name, acuity, complaint in specs]


_SAMPLE_SPECS = [
    ("Alice",   2, "sore throat"),
    ("Bob",     4, "abdominal pain"),
    ("Carol",   6, "palpitations"),
    ("Dave",    9, "chest pain — crushing"),
    ("Eve",    10, "cardiac arrest"),
]


# ---------------------------------------------------------------------------
# Admit
# ---------------------------------------------------------------------------

class TestAdmit:
    def test_returns_patient(self):
        reg = PatientRegistry()
        p = reg.admit("Alice", 5, "headache")
        assert isinstance(p, Patient)
        assert p.name == "Alice"
        assert p.acuity == 5

    def test_size_increments(self):
        reg = PatientRegistry()
        assert len(reg) == 0
        reg.admit("A", 3, "c")
        assert len(reg) == 1
        reg.admit("B", 7, "c")
        assert len(reg) == 2

    def test_is_admitted_after_admit(self):
        reg = PatientRegistry()
        p = reg.admit("Alice", 4, "pain")
        assert reg.is_admitted(p.patient_id)

    def test_admit_multiple_same_acuity(self):
        reg = PatientRegistry()
        p1 = reg.admit("A", 5, "c1")
        p2 = reg.admit("B", 5, "c2")
        assert reg.is_admitted(p1.patient_id)
        assert reg.is_admitted(p2.patient_id)
        assert len(reg) == 2

    def test_admit_patient_object(self):
        reg = PatientRegistry()
        p = Patient("Zara", 6, "seizure")
        reg.admit_patient(p)
        assert reg.is_admitted(p.patient_id)

    def test_duplicate_raises(self):
        reg = PatientRegistry()
        p = Patient("Duplicate", 5, "test")
        reg.admit_patient(p)
        with pytest.raises(DuplicateAdmissionError):
            reg.admit_patient(p)


# ---------------------------------------------------------------------------
# Discharge
# ---------------------------------------------------------------------------

class TestDischarge:
    def test_discharge_returns_patient(self):
        reg = PatientRegistry()
        p = reg.admit("Alice", 5, "headache")
        returned = reg.discharge(p.patient_id)
        assert returned == p

    def test_not_admitted_after_discharge(self):
        reg = PatientRegistry()
        p = reg.admit("Bob", 3, "bruise")
        reg.discharge(p.patient_id)
        assert not reg.is_admitted(p.patient_id)

    def test_size_decrements(self):
        reg = PatientRegistry()
        p = reg.admit("Carol", 7, "trauma")
        assert len(reg) == 1
        reg.discharge(p.patient_id)
        assert len(reg) == 0

    def test_discharge_unknown_raises(self):
        reg = PatientRegistry()
        with pytest.raises(PatientNotFoundError):
            reg.discharge("NOSUCHID")

    def test_discharge_one_of_many(self):
        reg = PatientRegistry()
        patients = _populate(reg, _SAMPLE_SPECS)
        target = patients[2]  # Carol, acuity=6
        reg.discharge(target.patient_id)
        assert not reg.is_admitted(target.patient_id)
        assert len(reg) == len(_SAMPLE_SPECS) - 1
        # Others unaffected
        for p in patients:
            if p.patient_id != target.patient_id:
                assert reg.is_admitted(p.patient_id)

    def test_discharge_all(self):
        reg = PatientRegistry()
        patients = _populate(reg, _SAMPLE_SPECS)
        for p in patients:
            reg.discharge(p.patient_id)
        assert len(reg) == 0


# ---------------------------------------------------------------------------
# Look-ups
# ---------------------------------------------------------------------------

class TestLookups:
    def test_get_patient(self):
        reg = PatientRegistry()
        p = reg.admit("Lin", 8, "stroke")
        fetched = reg.get_patient(p.patient_id)
        assert fetched == p

    def test_get_patient_missing_raises(self):
        reg = PatientRegistry()
        with pytest.raises(PatientNotFoundError):
            reg.get_patient("GHOST")

    def test_most_acute_single(self):
        reg = PatientRegistry()
        p = reg.admit("Solo", 7, "test")
        assert reg.most_acute() == p

    def test_most_acute_empty(self):
        reg = PatientRegistry()
        assert reg.most_acute() is None

    def test_most_acute_among_many(self):
        reg = PatientRegistry()
        _populate(reg, _SAMPLE_SPECS)
        top = reg.most_acute()
        assert top is not None
        assert top.acuity == 10   # Eve

    def test_least_acute_among_many(self):
        reg = PatientRegistry()
        _populate(reg, _SAMPLE_SPECS)
        bottom = reg.least_acute()
        assert bottom is not None
        assert bottom.acuity == 2  # Alice

    def test_all_patients_sorted(self):
        reg = PatientRegistry()
        _populate(reg, _SAMPLE_SPECS)
        acuities = [p.acuity for p in reg.all_patients()]
        assert acuities == sorted(acuities)


# ---------------------------------------------------------------------------
# Range queries
# ---------------------------------------------------------------------------

class TestCountInRange:
    def setup_method(self):
        self.reg = PatientRegistry()
        # Acuities present: 2, 4, 6, 9, 10
        _populate(self.reg, _SAMPLE_SPECS)

    def test_count_full_range(self):
        assert self.reg.count_in_range(1, 10) == len(_SAMPLE_SPECS)

    def test_count_interior_range(self):
        # Range [3, 7] spans the middle band — Bob(4) and Carol(6)
        assert self.reg.count_in_range(3, 7) == 2

    def test_count_high_end(self):
        # Range [7, 10] covers the critical tier — Dave(9) and Eve(10)
        assert self.reg.count_in_range(7, 10) == 2

    def test_count_low_end(self):
        # Range [1, 2] contains only Alice(2)
        assert self.reg.count_in_range(1, 2) == 1

    def test_count_empty_range(self):
        # No patient has acuity 7; range [7, 7] should return 0
        assert self.reg.count_in_range(7, 7) == 0

    def test_count_inverted_range(self):
        assert self.reg.count_in_range(8, 5) == 0

    def test_patients_in_range_content(self):
        patients = self.reg.patients_in_range(4, 6)
        acuities = {p.acuity for p in patients}
        assert acuities == {4, 6}

    def test_patients_in_range_sorted(self):
        patients = self.reg.patients_in_range(1, 10)
        acuities = [p.acuity for p in patients]
        assert acuities == sorted(acuities)


# ---------------------------------------------------------------------------
# Ward partitioning
# ---------------------------------------------------------------------------

class TestTransferAbove:
    def setup_method(self):
        self.reg = PatientRegistry()
        # Acuities: 2, 4, 6, 9, 10
        _populate(self.reg, _SAMPLE_SPECS)

    def test_transfer_above_yields_correct_size(self):
        # Threshold=7 → patients with acuity >= 7 should transfer (9, 10)
        # No patient has acuity exactly 7, so no boundary ambiguity
        transferred = self.reg.transfer_above(7)
        assert len(transferred) == 2

    def test_transfer_above_remaining_size(self):
        self.reg.transfer_above(7)
        assert len(self.reg) == 3  # Alice(2), Bob(4), Carol(6)

    def test_transfer_above_acuities_correct(self):
        transferred = self.reg.transfer_above(7)
        for p in transferred:
            assert p.acuity >= 7
        for p in self.reg:
            assert p.acuity < 7

    def test_transfer_above_all(self):
        transferred = self.reg.transfer_above(1)
        assert len(transferred) == len(_SAMPLE_SPECS)
        assert len(self.reg) == 0

    def test_transfer_above_none(self):
        transferred = self.reg.transfer_above(11)
        assert len(transferred) == 0
        assert len(self.reg) == len(_SAMPLE_SPECS)

    def test_absorb_restores_size(self):
        transferred = self.reg.transfer_above(7)
        original_size = len(self.reg) + len(transferred)
        self.reg.absorb(transferred)
        assert len(self.reg) == original_size


# ---------------------------------------------------------------------------
# Iteration and bulk access
# ---------------------------------------------------------------------------

class TestBulkAccess:
    def test_iter_yields_all(self):
        reg = PatientRegistry()
        patients = _populate(reg, _SAMPLE_SPECS)
        ids_from_iter = {p.patient_id for p in reg}
        ids_from_spec = {p.patient_id for p in patients}
        assert ids_from_iter == ids_from_spec

    def test_bool_empty(self):
        reg = PatientRegistry()
        assert not reg

    def test_bool_non_empty(self):
        reg = PatientRegistry()
        reg.admit("A", 5, "c")
        assert reg

    def test_critical_patients_filter(self):
        reg = PatientRegistry()
        _populate(reg, _SAMPLE_SPECS)
        critical = reg.critical_patients()
        for p in critical:
            assert p.acuity >= 8
