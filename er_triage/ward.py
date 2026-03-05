"""Ward and EmergencyDepartment models.

A Ward wraps a PatientRegistry and adds bed-management metadata.
An EmergencyDepartment composes two wards (general + ICU) and coordinates
patient flow between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .patient import Patient
from .registry import PatientNotFoundError, PatientRegistry


# ---------------------------------------------------------------------------
# Ward
# ---------------------------------------------------------------------------


@dataclass
class WardConfig:
    """Capacity and policy configuration for a ward."""

    name: str
    capacity: int = 50
    min_acuity: int = 1   # inclusive lower bound for acuity accepted here
    max_acuity: int = 10  # inclusive upper bound


class Ward:
    """A named ward that holds a set of admitted patients.

    Attributes
    ----------
    config:    WardConfig specifying name, capacity, and acuity policy.
    registry:  The underlying PatientRegistry.
    opened_at: Timestamp when the ward was initialised.
    """

    def __init__(self, config: WardConfig) -> None:
        self.config: WardConfig = config
        self.registry: PatientRegistry = PatientRegistry(label=config.name)
        self.opened_at: datetime = datetime.now()

    # ------------------------------------------------------------------
    # Patient flow
    # ------------------------------------------------------------------

    def admit(
        self,
        name: str,
        acuity: int,
        chief_complaint: str,
        attending: Optional[str] = None,
        notes: str = "",
    ) -> Patient:
        """Admit a new patient to this ward and return the Patient record."""
        if len(self.registry) >= self.config.capacity:
            raise RuntimeError(
                f"Ward '{self.config.name}' is at full capacity "
                f"({self.config.capacity} patients)."
            )
        return self.registry.admit(
            name=name,
            acuity=acuity,
            chief_complaint=chief_complaint,
            attending=attending,
            notes=notes,
        )

    def receive(self, patient: Patient) -> None:
        """Accept a patient transferred from another ward."""
        self.registry.admit_patient(patient)

    def discharge(self, patient_id: str) -> Patient:
        """Discharge a patient and return their record."""
        return self.registry.discharge(patient_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_occupied(self, patient_id: str) -> bool:
        return self.registry.is_admitted(patient_id)

    def occupancy(self) -> int:
        return len(self.registry)

    def occupancy_pct(self) -> float:
        return 100.0 * self.occupancy() / self.config.capacity

    def all_patients(self) -> List[Patient]:
        return self.registry.all_patients()

    def most_acute(self) -> Optional[Patient]:
        return self.registry.most_acute()

    def count_in_range(self, lo: int, hi: int) -> int:
        return self.registry.count_in_range(lo, hi)

    def __len__(self) -> int:
        return self.occupancy()

    def __repr__(self) -> str:
        return (
            f"Ward({self.config.name!r}, "
            f"occupancy={self.occupancy()}/{self.config.capacity})"
        )


# ---------------------------------------------------------------------------
# EmergencyDepartment
# ---------------------------------------------------------------------------


class EmergencyDepartment:
    """Two-tier emergency department composed of a general ward and an ICU.

    The general ward accepts all incoming patients.  The ICU receives patients
    who are escalated above an acuity threshold via escalate_critical().

    Example
    -------
    >>> dept = EmergencyDepartment("County General")
    >>> dept.admit("Bob", acuity=9, chief_complaint="cardiac arrest")
    >>> dept.admit("Sue", acuity=2, chief_complaint="minor laceration")
    >>> dept.escalate_critical(threshold=8)
    >>> dept.icu.occupancy()
    1
    >>> dept.general.occupancy()
    1
    """

    def __init__(
        self,
        name: str,
        general_capacity: int = 40,
        icu_capacity: int = 10,
    ) -> None:
        self.name: str = name
        self.general: Ward = Ward(
            WardConfig(name=f"{name} — General", capacity=general_capacity)
        )
        self.icu: Ward = Ward(
            WardConfig(
                name=f"{name} — ICU",
                capacity=icu_capacity,
                min_acuity=8,
            )
        )
        self._escalation_log: list[tuple[datetime, int, int]] = []

    # ------------------------------------------------------------------
    # Patient flow
    # ------------------------------------------------------------------

    def admit(
        self,
        name: str,
        acuity: int,
        chief_complaint: str,
        attending: Optional[str] = None,
        notes: str = "",
    ) -> Patient:
        """Admit a new patient to the general ward."""
        return self.general.admit(
            name=name,
            acuity=acuity,
            chief_complaint=chief_complaint,
            attending=attending,
            notes=notes,
        )

    def discharge(self, patient_id: str) -> Patient:
        """Discharge a patient from whichever ward they currently occupy."""
        for ward in (self.general, self.icu):
            if ward.is_occupied(patient_id):
                return ward.discharge(patient_id)
        raise PatientNotFoundError(patient_id)

    def escalate_critical(self, threshold: int = 8) -> int:
        """Move all patients with acuity >= threshold from general to ICU.

        Returns the number of patients transferred.
        """
        before = self.icu.occupancy()
        transferred_registry = self.general.registry.transfer_above(threshold)
        for patient in transferred_registry.all_patients():
            self.icu.receive(patient)
        after = self.icu.occupancy()
        n = after - before
        self._escalation_log.append((datetime.now(), threshold, n))
        return n

    def locate(self, patient_id: str) -> Optional[str]:
        """Return the ward name where the patient is located, or None."""
        if self.general.is_occupied(patient_id):
            return self.general.config.name
        if self.icu.is_occupied(patient_id):
            return self.icu.config.name
        return None

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    def total_occupancy(self) -> int:
        return self.general.occupancy() + self.icu.occupancy()

    def all_patients(self) -> List[Patient]:
        """All patients across both wards, sorted by acuity ascending."""
        combined = self.general.all_patients() + self.icu.all_patients()
        return sorted(combined, key=lambda p: p.acuity)

    def critical_count(self, threshold: int = 8) -> int:
        """Number of patients across both wards with acuity strictly above threshold."""
        return (
            self.general.registry.count_in_range(threshold + 1, 10)
            + self.icu.registry.count_in_range(threshold + 1, 10)
        )

    def census(self) -> str:
        """Return a multi-line census string for both wards."""
        lines = [
            f"=== {self.name} Census ===",
            f"General : {self.general.occupancy():>3} / {self.general.config.capacity}",
            f"ICU     : {self.icu.occupancy():>3} / {self.icu.config.capacity}",
            f"Total   : {self.total_occupancy():>3}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"EmergencyDepartment({self.name!r}, "
            f"general={self.general.occupancy()}, icu={self.icu.occupancy()})"
        )
