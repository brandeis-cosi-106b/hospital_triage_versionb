"""Tests for the Patient data model."""

import pytest
from er_triage.patient import Patient, acuity_label, acuity_category


# ---------------------------------------------------------------------------
# acuity_label / acuity_category helpers
# ---------------------------------------------------------------------------

class TestAcuityHelpers:
    def test_label_boundaries(self):
        assert acuity_label(1)  == "Non-urgent"
        assert acuity_label(10) == "Immediate"

    def test_label_mid(self):
        assert acuity_label(5) == "High"
        assert acuity_label(8) == "Critical"

    def test_label_unknown(self):
        assert acuity_label(0)  == "Unknown"
        assert acuity_label(11) == "Unknown"

    def test_category_green(self):
        for score in (1, 2, 3):
            assert acuity_category(score) == "Green"

    def test_category_yellow(self):
        for score in (4, 5, 6):
            assert acuity_category(score) == "Yellow"

    def test_category_orange(self):
        for score in (7, 8):
            assert acuity_category(score) == "Orange"

    def test_category_red(self):
        for score in (9, 10):
            assert acuity_category(score) == "Red"


# ---------------------------------------------------------------------------
# Patient construction
# ---------------------------------------------------------------------------

class TestPatientConstruction:
    def test_basic_fields(self):
        p = Patient(name="Alice", acuity=5, chief_complaint="headache")
        assert p.name == "Alice"
        assert p.acuity == 5
        assert p.chief_complaint == "headache"

    def test_patient_id_generated(self):
        p1 = Patient("A", 3, "cough")
        p2 = Patient("B", 3, "cough")
        assert p1.patient_id != p2.patient_id
        assert len(p1.patient_id) == 8

    def test_invalid_acuity_low(self):
        with pytest.raises(ValueError):
            Patient("X", 0, "test")

    def test_invalid_acuity_high(self):
        with pytest.raises(ValueError):
            Patient("X", 11, "test")

    def test_invalid_acuity_float(self):
        with pytest.raises((ValueError, TypeError)):
            Patient("X", 4.5, "test")  # type: ignore

    def test_empty_name(self):
        with pytest.raises(ValueError):
            Patient("  ", 5, "test")

    def test_attending_default_none(self):
        p = Patient("Bob", 2, "bruise")
        assert p.attending is None

    def test_notes_default_empty(self):
        p = Patient("Carol", 7, "chest pain")
        assert p.notes == ""

    def test_optional_fields_set(self):
        p = Patient(
            name="Dave", acuity=9, chief_complaint="cardiac arrest",
            attending="Dr. Kim", notes="STEMI protocol initiated"
        )
        assert p.attending == "Dr. Kim"
        assert "STEMI" in p.notes


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------

class TestPatientProperties:
    def test_severity_label(self):
        p = Patient("E", 8, "test")
        assert p.severity_label == "Critical"

    def test_triage_category_red(self):
        p = Patient("F", 9, "test")
        assert p.triage_category == "Red"

    def test_triage_category_green(self):
        p = Patient("G", 2, "test")
        assert p.triage_category == "Green"

    def test_is_critical_true(self):
        for score in (8, 9, 10):
            assert Patient("H", score, "t").is_critical

    def test_is_critical_false(self):
        for score in (1, 2, 3, 4, 5, 6, 7):
            assert not Patient("H", score, "t").is_critical

    def test_wait_minutes_positive(self):
        p = Patient("I", 3, "test")
        assert p.wait_minutes >= 0.0


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

class TestPatientMutation:
    def test_add_note_prepends(self):
        p = Patient("J", 4, "test")
        p.add_note("First note")
        p.add_note("Second note")
        assert p.notes.index("Second") < p.notes.index("First")

    def test_assign_attending(self):
        p = Patient("K", 6, "test")
        p.assign_attending("Dr. Reyes")
        assert p.attending == "Dr. Reyes"

    def test_assign_attending_overwrite(self):
        p = Patient("L", 6, "test")
        p.assign_attending("Dr. A")
        p.assign_attending("Dr. B")
        assert p.attending == "Dr. B"


# ---------------------------------------------------------------------------
# Equality and hashing
# ---------------------------------------------------------------------------

class TestPatientEquality:
    def test_same_id_equal(self):
        p = Patient("M", 3, "test")
        # Reconstruct with same ID
        q = Patient("M", 3, "test", patient_id=p.patient_id)
        assert p == q

    def test_different_id_not_equal(self):
        p1 = Patient("N", 3, "test")
        p2 = Patient("N", 3, "test")
        assert p1 != p2

    def test_hashable_in_set(self):
        p = Patient("O", 5, "test")
        s = {p}
        assert p in s

    def test_not_equal_to_non_patient(self):
        p = Patient("P", 5, "test")
        assert p != "not a patient"
