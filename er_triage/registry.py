"""PatientRegistry — the primary public interface for managing ER patients."""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, List, Optional

from .patient import Patient
from ._store import _AcuityIndex


class RegistryError(Exception):
    """Raised for invalid operations on the registry."""


class PatientNotFoundError(RegistryError):
    """Raised when a requested patient_id does not exist in the registry."""

    def __init__(self, patient_id: str) -> None:
        super().__init__(f"No patient with ID {patient_id!r} is currently registered.")
        self.patient_id = patient_id


class DuplicateAdmissionError(RegistryError):
    """Raised when attempting to admit a patient who is already registered."""

    def __init__(self, patient_id: str) -> None:
        super().__init__(
            f"Patient {patient_id!r} is already admitted.  "
            "Discharge them first before re-admitting."
        )
        self.patient_id = patient_id


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PatientRegistry:
    """Manages a collection of admitted patients for a single ward.

    Patients are indexed by acuity score to support fast range queries and
    threshold-based partitioning.  All mutations are O(log n) expected;
    look-ups by patient_id are O(n) (full scan).

    Example
    -------
    >>> reg = PatientRegistry()
    >>> alice = reg.admit("Alice Nguyen", acuity=6, chief_complaint="severe headache")
    >>> reg.is_admitted(alice.patient_id)
    True
    >>> reg.most_acute()
    Patient(name='Alice Nguyen', acuity=6, id=...)
    >>> reg.discharge(alice.patient_id)
    >>> reg.is_admitted(alice.patient_id)
    False
    """

    def __init__(self, label: str = "") -> None:
        self._index: _AcuityIndex = _AcuityIndex()
        self.label: str = label
        self._admission_log: list[tuple[datetime, str, str]] = []  # (time, event, pid)

    # ------------------------------------------------------------------
    # Core admission / discharge
    # ------------------------------------------------------------------

    def admit(
        self,
        name: str,
        acuity: int,
        chief_complaint: str,
        attending: Optional[str] = None,
        notes: str = "",
    ) -> Patient:
        """Create and register a new patient.  Returns the Patient record.

        Raises DuplicateAdmissionError if a patient with the same auto-generated
        ID already exists (astronomically unlikely in practice).
        """
        patient = Patient(
            name=name,
            acuity=acuity,
            chief_complaint=chief_complaint,
            attending=attending,
            notes=notes,
        )
        self._index.insert(patient)
        self._log("ADMIT", patient.patient_id)
        return patient

    def admit_patient(self, patient: Patient) -> None:
        """Register an already-constructed Patient object.

        Use this when transferring a patient from another registry so that
        the original Patient record (and its ID) are preserved.
        """
        if self._index.contains(patient.patient_id):
            raise DuplicateAdmissionError(patient.patient_id)
        self._index.insert(patient)
        self._log("ADMIT", patient.patient_id)

    def discharge(self, patient_id: str) -> Patient:
        """Remove a patient from the registry and return their record.

        Raises PatientNotFoundError if the patient is not currently admitted.
        """
        patient = self._index.find(patient_id)
        if patient is None:
            raise PatientNotFoundError(patient_id)
        self._index.remove(patient_id)
        self._log("DISCHARGE", patient_id)
        return patient

    # ------------------------------------------------------------------
    # Look-ups
    # ------------------------------------------------------------------

    def is_admitted(self, patient_id: str) -> bool:
        """Return True if the patient is currently in this registry."""
        return self._index.contains(patient_id)

    def get_patient(self, patient_id: str) -> Patient:
        """Return the Patient record for the given ID.

        Raises PatientNotFoundError if absent.
        """
        patient = self._index.find(patient_id)
        if patient is None:
            raise PatientNotFoundError(patient_id)
        return patient

    def most_acute(self) -> Optional[Patient]:
        """Return the patient with the highest acuity score, or None if empty."""
        return self._index.maximum()

    def least_acute(self) -> Optional[Patient]:
        """Return the patient with the lowest acuity score, or None if empty."""
        return self._index.minimum()

    # ------------------------------------------------------------------
    # Range queries
    # ------------------------------------------------------------------

    def count_in_range(self, lo_acuity: int, hi_acuity: int) -> int:
        """Return the number of patients with acuity in [lo_acuity, hi_acuity].

        Both endpoints are inclusive.  Returns 0 if lo_acuity > hi_acuity.
        """
        return self._index.count_in_range(lo_acuity, hi_acuity)

    def patients_in_range(self, lo_acuity: int, hi_acuity: int) -> List[Patient]:
        """Return all patients whose acuity falls in [lo_acuity, hi_acuity],
        sorted in ascending acuity order."""
        return [p for p in self._index.inorder() if lo_acuity <= p.acuity <= hi_acuity]

    # ------------------------------------------------------------------
    # Ward-level partitioning
    # ------------------------------------------------------------------

    def transfer_above(self, acuity_threshold: int) -> "PatientRegistry":
        """Move all patients with acuity >= acuity_threshold out of this registry.

        Returns a new PatientRegistry containing the transferred patients.
        The current registry is modified in-place to retain only those with
        acuity strictly below the threshold.
        """
        extracted_index = self._index.split_above(acuity_threshold)
        transferred = PatientRegistry(label=f"transfer@{acuity_threshold}+")
        transferred._index = extracted_index
        self._log("TRANSFER_OUT", f"threshold={acuity_threshold}")
        return transferred

    def absorb(self, other: "PatientRegistry") -> None:
        """Merge all patients from *other* into this registry.

        Intended for reuniting two registries that were split with
        transfer_above().  The *other* registry is emptied.
        """
        self._index.merge_from(other._index)
        self._log("ABSORB", f"from={other.label!r}")

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    def all_patients(self) -> List[Patient]:
        """Return all patients sorted by acuity (ascending)."""
        return self._index.inorder()

    def critical_patients(self) -> List[Patient]:
        """Return patients with acuity >= 8, sorted ascending."""
        return [p for p in self._index.inorder() if p.acuity >= 8]

    def __len__(self) -> int:
        return len(self._index)

    def __bool__(self) -> bool:
        return bool(self._index)

    def __iter__(self) -> Iterator[Patient]:
        return iter(self._index)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, event: str, detail: str) -> None:
        self._admission_log.append((datetime.now(), event, detail))

    def __repr__(self) -> str:
        return f"PatientRegistry(label={self.label!r}, size={len(self)})"
