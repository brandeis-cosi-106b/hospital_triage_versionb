"""
ER Triage Registry — single-file version.

This file is a flat concatenation of the er_triage package, the simulation
harness, and the test suite.  It is provided for environments where copying
and pasting a multi-file project is impractical.

Run the tests with:
    pip install pytest
    pytest onelargefile.py
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import pytest


# ===========================================================================
# SECTION 1: Patient Data Model  (er_triage/patient.py)
# ===========================================================================


"""Patient data model for the ER Triage Registry."""



# ---------------------------------------------------------------------------
# Acuity helpers
# ---------------------------------------------------------------------------

_ACUITY_LABELS: dict[int, str] = {
    1:  "Non-urgent",
    2:  "Low",
    3:  "Moderate",
    4:  "Urgent",
    5:  "High",
    6:  "Very High",
    7:  "Severe",
    8:  "Critical",
    9:  "Life-threatening",
    10: "Immediate",
}

# Colour-coded triage categories (Green / Yellow / Orange / Red)
_CATEGORY_BOUNDS: list[tuple[range, str]] = [
    (range(1, 4),  "Green"),
    (range(4, 7),  "Yellow"),
    (range(7, 9),  "Orange"),
    (range(9, 11), "Red"),
]


def acuity_label(acuity: int) -> str:
    """Return the severity label for a given acuity score (1–10)."""
    return _ACUITY_LABELS.get(acuity, "Unknown")


def acuity_category(acuity: int) -> str:
    """Return the triage colour category for a given acuity score."""
    for r, cat in _CATEGORY_BOUNDS:
        if acuity in r:
            return cat
    return "Unknown"


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    """Represents a single patient in the ER triage system.

    Attributes
    ----------
    name:              Full name of the patient.
    acuity:            Clinical severity score, 1 (non-urgent) to 10 (immediate).
    chief_complaint:   Primary presenting symptom or reason for visit.
    patient_id:        Unique 8-character identifier assigned at admission.
    arrival_time:      Timestamp when the patient arrived.
    notes:             Free-text clinical notes accumulated during the visit.
    attending:         Name of the assigned attending physician, if any.
    """

    name: str
    acuity: int
    chief_complaint: str
    patient_id: str = field(
        default_factory=lambda: str(uuid.uuid4()).replace("-", "")[:8].upper()
    )
    arrival_time: datetime = field(default_factory=datetime.now)
    notes: str = ""
    attending: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Patient name cannot be empty.")
        if not isinstance(self.acuity, int) or not (1 <= self.acuity <= 10):
            raise ValueError(
                f"Acuity must be an integer between 1 and 10, got {self.acuity!r}."
            )

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def severity_label(self) -> str:
        """Human-readable severity string (e.g. 'Critical')."""
        return acuity_label(self.acuity)

    @property
    def triage_category(self) -> str:
        """Colour-coded triage category (Green / Yellow / Orange / Red)."""
        return acuity_category(self.acuity)

    @property
    def is_critical(self) -> bool:
        """True if the patient has acuity ≥ 8."""
        return self.acuity >= 8

    @property
    def wait_minutes(self) -> float:
        """Elapsed time in minutes since arrival."""
        return (datetime.now() - self.arrival_time).total_seconds() / 60.0

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_note(self, note: str) -> None:
        """Prepend a timestamped note to the patient's record."""
        ts = datetime.now().strftime("%H:%M")
        self.notes = f"[{ts}] {note}\n" + self.notes

    def assign_attending(self, physician_name: str) -> None:
        """Assign or reassign the attending physician."""
        self.attending = physician_name

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary_line(self) -> str:
        """Single-line summary suitable for console output."""
        attended = f" | Attending: {self.attending}" if self.attending else ""
        return (
            f"[{self.patient_id}] {self.name:<20} "
            f"Acuity {self.acuity:>2} ({self.severity_label:<16}) "
            f"CC: {self.chief_complaint}{attended}"
        )

    def __str__(self) -> str:
        return self.summary_line()

    def __repr__(self) -> str:
        return (
            f"Patient(name={self.name!r}, acuity={self.acuity}, "
            f"id={self.patient_id!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Patient):
            return NotImplemented
        return self.patient_id == other.patient_id

    def __hash__(self) -> int:
        return hash(self.patient_id)

# ===========================================================================
# SECTION 2: Internal Patient Store  (er_triage/_store.py)
# ===========================================================================


"""
Internal patient storage layer.

Provides the backing store used by PatientRegistry to maintain patients and
support fast admission, discharge, range counting, and ward partitioning.

This module is private to the er_triage package.  Consumers should interact
only with PatientRegistry and Ward, which provide the stable public interface.
"""




# ---------------------------------------------------------------------------
# Internal tree node
# ---------------------------------------------------------------------------


class _Node:
    """A single node in the acuity-sorted index tree.

    Each node stores one patient, a randomly-assigned structural key used to
    maintain balance, and left/right child pointers.
    """

    __slots__ = ("patient", "_balance_key", "left", "right")

    def __init__(self, patient: Patient) -> None:
        self.patient: Patient = patient
        # The balance key is assigned once at insertion and never changes.
        # It drives the heap-like structural invariant that keeps the tree
        # balanced in expectation without explicit rebalancing passes.
        self._balance_key: float = random.random()
        self.left: Optional[_Node] = None
        self.right: Optional[_Node] = None

    # Convenience aliases so callers can read node.acuity / node.patient_id
    # directly without going through node.patient each time.

    @property
    def acuity(self) -> int:
        return self.patient.acuity

    @property
    def patient_id(self) -> str:
        return self.patient.patient_id

    def __repr__(self) -> str:
        return f"_Node(acuity={self.acuity}, id={self.patient_id!r})"


# ---------------------------------------------------------------------------
# Patient store
# ---------------------------------------------------------------------------


class _AcuityIndex:
    """In-memory patient store ordered by acuity.

    Patients are kept in acuity order to support efficient range queries and
    threshold partitioning.  The structural balance key stored in each node
    keeps the expected height at O(log n) without explicit rebalancing.

    Supported operations
    --------------------
    insert(patient)               O(log n) expected
    remove(patient_id)            O(n) search + O(log n) removal
    contains(patient_id)          O(n)
    find(patient_id)              O(n)
    maximum() / minimum()         O(log n)
    count_in_range(lo, hi)        O(log n)
    split_above(threshold)        O(log n)  — destructive partition
    merge_from(other)             O(log n)  — absorb another index
    inorder()                     O(n)
    """

    def __init__(self) -> None:
        self._root: Optional[_Node] = None
        self._count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def insert(self, patient: Patient) -> None:
        """Add *patient* to the index."""
        self._root = self._insert(self._root, patient)
        self._count += 1

    def remove(self, patient_id: str) -> bool:
        """Remove the patient with the given ID.  Returns True if found."""
        new_root, removed = self._remove(self._root, patient_id)
        if removed:
            self._root = new_root
            self._count -= 1
        return removed

    def contains(self, patient_id: str) -> bool:
        """Return True if a patient with *patient_id* is in the index."""
        return self._find_node(self._root, patient_id) is not None

    def find(self, patient_id: str) -> Optional[Patient]:
        """Return the Patient with the given ID, or None if absent."""
        node = self._find_node(self._root, patient_id)
        return node.patient if node else None

    def maximum(self) -> Optional[Patient]:
        """Return the patient with the highest acuity score, or None."""
        node = self._rightmost(self._root)
        return node.patient if node else None

    def minimum(self) -> Optional[Patient]:
        """Return the patient with the lowest acuity score, or None."""
        node = self._leftmost(self._root)
        return node.patient if node else None

    def count_in_range(self, lo: int, hi: int) -> int:
        """Count patients whose acuity is in the closed interval [lo, hi].

        The index is temporarily restructured during the computation and
        fully restored before returning.  Behaviour is undefined if lo > hi.
        """
        if lo > hi:
            return 0
        # Carve out the [lo, hi] band with two partitions then count its size.
        left, mid_and_right = self._partition(self._root, lo)
        mid, right = self._partition(mid_and_right, hi + 1)
        result = self._subtree_size(mid)
        # Restore original structure.
        self._root = self._merge(self._merge(left, mid), right)
        return result

    def split_above(self, acuity_threshold: int) -> "_AcuityIndex":
        """Destructively partition: remove and return patients with acuity >= threshold.

        After this call the current index retains only patients whose acuity
        is strictly below *acuity_threshold*.  The returned index holds all
        patients at or above the threshold.
        """
        below, above = self._partition(self._root, acuity_threshold)
        self._root = below
        self._count = self._subtree_size(below)
        extracted = _AcuityIndex()
        extracted._root = above
        extracted._count = self._subtree_size(above)
        return extracted

    def merge_from(self, other: "_AcuityIndex") -> None:
        """Absorb all patients from *other* into this index.

        Precondition: every patient in *other* must have strictly higher acuity
        than every patient currently in this index, or vice-versa.  Callers
        are responsible for ensuring this invariant.
        """
        self._root = self._merge(self._root, other._root)
        self._count += other._count
        other._root = None
        other._count = 0

    def inorder(self) -> List[Patient]:
        """Return all patients sorted by acuity (ascending)."""
        result: List[Patient] = []
        self._collect_inorder(self._root, result)
        return result

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def __iter__(self) -> Iterator[Patient]:
        return iter(self.inorder())

    # ------------------------------------------------------------------
    # Core structural primitives
    # ------------------------------------------------------------------

    def _partition(
        self, node: Optional[_Node], threshold: int
    ) -> Tuple[Optional[_Node], Optional[_Node]]:
        """Split the subtree rooted at *node* into two disjoint subtrees.

        Returns (lower, upper) where:
            lower — patients with acuity <  threshold
            upper — patients with acuity >= threshold
        """
        if node is None:
            return None, None
        if node.acuity < threshold:
            lower, upper = self._partition(node.right, threshold)
            node.right = lower
            return node, upper
        else:
            lower, upper = self._partition(node.left, threshold)
            node.left = upper
            return lower, node

    def _merge(
        self, left: Optional[_Node], right: Optional[_Node]
    ) -> Optional[_Node]:
        """Merge two subtrees where every key in *left* is less than every key
        in *right*.  The balance keys determine the merged tree's shape."""
        if left is None:
            return right
        if right is None:
            return left
        if left._balance_key < right._balance_key:
            left.right = self._merge(left.right, right)
            return left
        else:
            right.left = self._merge(left, right.left)
            return right

    def _insert(self, node: Optional[_Node], patient: Patient) -> _Node:
        """Insert *patient* and return the new subtree root."""
        new_node = _Node(patient)
        lower, upper = self._partition(node, patient.acuity)
        return self._merge(self._merge(lower, new_node), upper)

    def _remove(
        self, node: Optional[_Node], patient_id: str
    ) -> Tuple[Optional[_Node], bool]:
        """Search for *patient_id* and remove it.  Returns (new_root, found)."""
        if node is None:
            return None, False
        if node.patient_id == patient_id:
            # Splice out by merging the two children directly.
            return self._merge(node.left, node.right), True
        # The tree is ordered by acuity, not patient_id, so we must check
        # both subtrees.  Try left first; fall through to right if not found.
        new_left, found = self._remove(node.left, patient_id)
        if found:
            node.left = new_left
            return node, True
        new_right, found = self._remove(node.right, patient_id)
        node.right = new_right
        return node, found

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def _find_node(
        self, node: Optional[_Node], patient_id: str
    ) -> Optional[_Node]:
        """Full-tree search for a node by patient_id (O(n))."""
        if node is None:
            return None
        if node.patient_id == patient_id:
            return node
        found = self._find_node(node.left, patient_id)
        if found is not None:
            return found
        return self._find_node(node.right, patient_id)

    @staticmethod
    def _leftmost(node: Optional[_Node]) -> Optional[_Node]:
        if node is None:
            return None
        while node.left is not None:
            node = node.left
        return node

    @staticmethod
    def _rightmost(node: Optional[_Node]) -> Optional[_Node]:
        if node is None:
            return None
        while node.right is not None:
            node = node.right
        return node

    @staticmethod
    def _subtree_size(node: Optional[_Node]) -> int:
        if node is None:
            return 0
        return 1 + _AcuityIndex._subtree_size(node.left) + _AcuityIndex._subtree_size(node.right)

    @staticmethod
    def _collect_inorder(node: Optional[_Node], out: List[Patient]) -> None:
        if node is None:
            return
        _AcuityIndex._collect_inorder(node.left, out)
        out.append(node.patient)
        _AcuityIndex._collect_inorder(node.right, out)

# ===========================================================================
# SECTION 3: Patient Registry  (er_triage/registry.py)
# ===========================================================================


"""PatientRegistry — the primary public interface for managing ER patients."""





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

# ===========================================================================
# SECTION 4: Ward and Emergency Dept  (er_triage/ward.py)
# ===========================================================================


"""Ward and EmergencyDepartment models.

A Ward wraps a PatientRegistry and adds bed-management metadata.
An EmergencyDepartment composes two wards (general + ICU) and coordinates
patient flow between them.
"""





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

# ===========================================================================
# SECTION 5: Reports and Metrics  (er_triage/reports.py)
# ===========================================================================


"""Census and operational reporting for the ER Triage Registry.

All functions in this module are read-only with respect to the registry;
they produce formatted strings or plain-data summaries without mutating state.
"""






# ---------------------------------------------------------------------------
# Low-level formatting helpers
# ---------------------------------------------------------------------------

_DIVIDER = "-" * 72
_BOLD_DIVIDER = "=" * 72


def _header(title: str) -> str:
    pad = (_BOLD_DIVIDER.__len__() - len(title) - 2) // 2
    return f"{'=' * pad} {title} {'=' * pad}"


def _patient_row(p: Patient) -> str:
    waited = f"{p.wait_minutes:.0f}m"
    attending = p.attending or "—"
    return (
        f"  {p.patient_id:<10} {p.name:<22} "
        f"A{p.acuity:<3} {p.severity_label:<18} "
        f"{p.chief_complaint:<28} wait={waited:<6} {attending}"
    )


def format_patient_table(patients: List[Patient], title: str = "Patients") -> str:
    """Return a fixed-width table of patients sorted by acuity descending."""
    if not patients:
        return f"{title}: (none)"
    rows = [_header(title)]
    rows.append(
        f"  {'ID':<10} {'Name':<22} {'Ac':<4} {'Severity':<18} "
        f"{'Chief Complaint':<28} {'Wait':<6} Attending"
    )
    rows.append(_DIVIDER)
    for p in sorted(patients, key=lambda x: -x.acuity):
        rows.append(_patient_row(p))
    rows.append(_BOLD_DIVIDER)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Acuity distribution
# ---------------------------------------------------------------------------


def acuity_distribution(registry: "PatientRegistry") -> Dict[int, int]:
    """Return a dict mapping each acuity level (1–10) to the patient count.

    Uses a full inorder traversal rather than repeated range queries so that
    it runs in a single O(n) pass.
    """
    counts: Dict[int, int] = {i: 0 for i in range(1, 11)}
    for patient in registry.all_patients():
        counts[patient.acuity] += 1
    return counts


def format_acuity_histogram(registry: "PatientRegistry") -> str:
    """Render a simple ASCII histogram of the acuity distribution."""
    dist = acuity_distribution(registry)
    max_count = max(dist.values()) if any(dist.values()) else 1
    bar_width = 30
    lines = [_header("Acuity Distribution")]
    for level in range(10, 0, -1):
        count = dist[level]
        filled = int(bar_width * count / max_count) if max_count else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        label = acuity_label(level)
        lines.append(
            f"  {level:>2} {label:<18} [{bar}] {count:>3}"
        )
    lines.append(_BOLD_DIVIDER)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category summary
# ---------------------------------------------------------------------------


def category_summary(registry: "PatientRegistry") -> Dict[str, int]:
    """Aggregate patient counts by triage colour category."""
    totals: Dict[str, int] = {"Green": 0, "Yellow": 0, "Orange": 0, "Red": 0}
    for patient in registry:
        cat = acuity_category(patient.acuity)
        if cat in totals:
            totals[cat] += 1
    return totals


def format_category_summary(registry: "PatientRegistry") -> str:
    summary = category_summary(registry)
    lines = [_header("Triage Category Summary")]
    colours = {"Green": "🟢", "Yellow": "🟡", "Orange": "🟠", "Red": "🔴"}
    for cat, count in summary.items():
        icon = colours.get(cat, " ")
        lines.append(f"  {icon}  {cat:<8} {count:>4} patient(s)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ward-level reports
# ---------------------------------------------------------------------------


def ward_report(ward: "Ward") -> str:
    """Full report for a single ward including patient table and stats."""
    patients = ward.all_patients()
    lines = [
        format_patient_table(patients, title=ward.config.name),
        f"  Occupancy : {ward.occupancy():>3} / {ward.config.capacity} "
        f"({ward.occupancy_pct():.1f}%)",
    ]
    if patients:
        avg = sum(p.acuity for p in patients) / len(patients)
        lines.append(f"  Avg acuity: {avg:.2f}")
        lines.append(f"  Most acute: {ward.most_acute()}")
    return "\n".join(lines)


def department_report(dept: "EmergencyDepartment") -> str:
    """Comprehensive report spanning both wards of an EmergencyDepartment."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = [
        _bold_divider_line(f"{dept.name.upper()} — DEPARTMENT REPORT"),
        f"  Generated : {ts}",
        f"  Total census : {dept.total_occupancy()} patient(s)",
        "",
        ward_report(dept.general),
        "",
        ward_report(dept.icu),
        "",
        format_acuity_histogram(dept.general.registry),
        "",
        format_category_summary(dept.general.registry),
        _BOLD_DIVIDER,
    ]
    return "\n".join(sections)


def _bold_divider_line(title: str) -> str:
    return _header(title)


# ---------------------------------------------------------------------------
# Load metrics
# ---------------------------------------------------------------------------


def critical_load(dept: "EmergencyDepartment", threshold: int = 8) -> dict:
    """Return a dict of load metrics relative to *threshold*.

    Keys
    ----
    critical_count   : patients with acuity >= threshold (both wards combined)
    general_critical : critical patients still in the general ward
    icu_count        : total ICU occupancy
    icu_pct          : ICU occupancy as a percentage of ICU capacity
    """
    gen_critical = dept.general.registry.count_in_range(threshold, 10)
    icu_count = dept.icu.occupancy()
    return {
        "critical_count":   gen_critical + icu_count,
        "general_critical": gen_critical,
        "icu_count":        icu_count,
        "icu_pct":          100.0 * icu_count / dept.icu.config.capacity,
    }


def format_load_metrics(dept: "EmergencyDepartment", threshold: int = 8) -> str:
    metrics = critical_load(dept, threshold)
    lines = [
        _header(f"Load Metrics (threshold = {threshold})"),
        f"  Critical patients (acuity >= {threshold}) : {metrics['critical_count']:>3}",
        f"    In general ward                        : {metrics['general_critical']:>3}",
        f"    In ICU                                 : {metrics['icu_count']:>3}",
        f"  ICU occupancy                            : {metrics['icu_pct']:.1f}%",
    ]
    return "\n".join(lines)

# ===========================================================================
# SECTION 6: Simulation — Data Generator  (simulation/data_generator.py)
# ===========================================================================


"""Synthetic patient and scenario data generation.

Produces realistic-looking (but entirely fictional) patient records for use
in simulation runs and stress testing.
"""



# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Aaliyah", "Ahmed", "Ana", "Andre", "Beatriz", "Cameron", "Chioma",
    "Daniel", "Dmitri", "Elena", "Emeka", "Fatima", "Grace", "Hassan",
    "Isabel", "Jamal", "Ji-ho", "Kenji", "Layla", "Lena", "Lin",
    "Marco", "Maria", "Mei", "Miguel", "Nadia", "Nikolai", "Olumide",
    "Priya", "Rahul", "Rosa", "Samuel", "Sofia", "Tariq", "Tunde",
    "Valentina", "Wei", "Yuki", "Zara", "Zhao",
]

_LAST_NAMES = [
    "Adeyemi", "Andersen", "Bashir", "Castillo", "Chen", "da Silva",
    "Dubois", "Eriksson", "Ferreira", "Garcia", "Gupta", "Hansen",
    "Hashimoto", "Hernandez", "Ibrahim", "Jensen", "Johansson", "Kim",
    "Kumar", "Laurent", "Lee", "Lopez", "Martinez", "Mensah", "Moreira",
    "Mueller", "Nakamura", "Nguyen", "Okafor", "Oliveira", "Patel",
    "Petrov", "Rahman", "Reyes", "Rodriguez", "Santos", "Schmidt",
    "Singh", "Smith", "Svensson", "Tanaka", "Torres", "Virtanen",
    "Wang", "Weber", "Williams", "Yamamoto", "Zhang",
]

# ---------------------------------------------------------------------------
# Chief complaints by acuity band
# ---------------------------------------------------------------------------

_COMPLAINTS_BY_BAND: dict[str, List[str]] = {
    "low": [           # acuity 1–3
        "minor laceration — right forearm",
        "sprained ankle — no deformity",
        "sore throat, mild fever",
        "earache, 2 days duration",
        "mild rash — no breathing difficulty",
        "prescription renewal request",
        "insect bite, localised swelling",
        "minor burn — hot liquid, <5% BSA",
        "headache, no focal neurology",
        "low back pain, chronic",
        "nausea and vomiting, 1 day",
        "urinary frequency, mild dysuria",
        "conjunctivitis — discharge, no vision change",
        "dental pain, no facial swelling",
        "anxiety symptoms, no SI",
    ],
    "moderate": [      # acuity 4–6
        "abdominal pain — RLQ, onset 6 h",
        "chest pain — atypical, reproducible",
        "shortness of breath — mild, SpO2 95%",
        "head laceration — moderate bleeding, controlled",
        "fracture — distal radius, closed",
        "syncope — single episode, resolved",
        "hypertensive urgency — BP 175/110",
        "acute confusion — elderly patient",
        "allergic reaction — hives, no angioedema",
        "closed head injury — GCS 14",
        "palpitations — HR 140 bpm, haemodynamically stable",
        "severe vomiting and diarrhoea — moderate dehydration",
        "acute asthma — moderate, SpO2 93%",
        "flank pain, haematuria — ?renal colic",
        "cellulitis — lower leg, low-grade fever",
    ],
    "high": [          # acuity 7–10
        "chest pain — crushing, radiation to left arm",
        "stroke symptoms — facial droop, arm weakness, onset 40 min",
        "respiratory failure — SpO2 < 88% on room air",
        "major trauma — MVC, multiple injuries",
        "anaphylaxis — airway involvement",
        "sepsis — temperature 39.8 °C, HR 118, BP 88/60",
        "STEMI — ST elevation V2–V5",
        "status epilepticus — ongoing seizure",
        "aortic dissection — tearing back pain",
        "massive haemorrhage — post-traumatic",
        "altered consciousness — GCS 8",
        "eclampsia — BP 180/120, seizure",
        "tension pneumothorax — tracheal deviation",
        "meningitis — neck stiffness, petechial rash",
        "overdose — altered mental status, pinpoint pupils",
    ],
}

# Attending physician pool
_ATTENDINGS = [
    "Dr. Abramowitz", "Dr. Castillo", "Dr. Chen", "Dr. Fitzgerald",
    "Dr. Gupta", "Dr. Johansson", "Dr. Obi", "Dr. Petrov",
    "Dr. Rashid", "Dr. Santos", "Dr. Tanaka", "Dr. Voss",
]


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def random_name(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def random_complaint(acuity: int, rng: random.Random) -> str:
    if acuity <= 3:
        pool = _COMPLAINTS_BY_BAND["low"]
    elif acuity <= 6:
        pool = _COMPLAINTS_BY_BAND["moderate"]
    else:
        pool = _COMPLAINTS_BY_BAND["high"]
    return rng.choice(pool)


def random_acuity(rng: random.Random, mean: float = 4.5, spread: float = 2.5) -> int:
    """Sample an acuity score skewed toward the lower end (most ER visits are not critical)."""
    raw = rng.gauss(mean, spread)
    return max(1, min(10, round(raw)))


def random_patient_spec(
    rng: random.Random,
    forced_acuity: int | None = None,
) -> Tuple[str, int, str, str | None]:
    """Return (name, acuity, complaint, attending) for one synthetic patient."""
    acuity = forced_acuity if forced_acuity is not None else random_acuity(rng)
    name = random_name(rng)
    complaint = random_complaint(acuity, rng)
    attending = rng.choice(_ATTENDINGS) if rng.random() < 0.6 else None
    return name, acuity, complaint, attending


def generate_arrival_batch(
    n: int,
    seed: int | None = None,
    acuity_override: int | None = None,
) -> List[Tuple[str, int, str, str | None]]:
    """Generate *n* synthetic patient arrival records.

    Parameters
    ----------
    n:               Number of patients to generate.
    seed:            Optional random seed for reproducibility.
    acuity_override: If provided, every patient gets this acuity.
    """
    rng = random.Random(seed)
    return [random_patient_spec(rng, forced_acuity=acuity_override) for _ in range(n)]

# ===========================================================================
# SECTION 7: Simulation — Simulator  (simulation/simulator.py)
# ===========================================================================


"""Discrete-event ER simulation.

Simulates a stream of patient arrivals, acuity assessments, escalations, and
discharges over a configurable time window.  Produces a human-readable log and
a final department report.

Usage
-----
    python -m simulation.simulator                    # default 8-hour shift
    python -m simulation.simulator --hours 4 --seed 99
"""


import time

# Ensure the package root is on the path when running as __main__


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Tunable parameters for one simulation run."""

    seed: int = 42
    shift_hours: float = 8.0
    arrival_rate_per_hour: float = 12.0   # patients per simulated hour
    discharge_rate_per_hour: float = 10.0  # discharges per simulated hour
    escalation_threshold: int = 8
    escalation_check_every_n_arrivals: int = 5
    general_capacity: int = 40
    icu_capacity: int = 10
    dept_name: str = "Simulation Hospital ER"
    verbose: bool = True


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single simulation event (arrival or discharge)."""

    timestamp: datetime
    kind: str           # "arrive" | "discharge" | "escalate" | "snapshot"
    detail: str = ""

    def log_line(self) -> str:
        ts = self.timestamp.strftime("%H:%M:%S")
        return f"[{ts}] {self.kind.upper():<10} {self.detail}"


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class ERSimulator:
    """Runs a discrete-event simulation of the emergency department."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.cfg = config or SimConfig()
        self.rng = random.Random(self.cfg.seed)
        self.dept = EmergencyDepartment(
            name=self.cfg.dept_name,
            general_capacity=self.cfg.general_capacity,
            icu_capacity=self.cfg.icu_capacity,
        )
        self._events: List[Event] = []
        self._admitted_ids: List[str] = []

    def _log(self, kind: str, detail: str, ts: datetime) -> None:
        ev = Event(timestamp=ts, kind=kind, detail=detail)
        self._events.append(ev)
        if self.cfg.verbose:
            print(ev.log_line())

    def _simulated_timestamps(self) -> List[datetime]:
        """Generate Poisson-distributed arrival times over the shift."""
        base = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        total_seconds = int(self.cfg.shift_hours * 3600)
        expected = int(self.cfg.arrival_rate_per_hour * self.cfg.shift_hours)
        times = sorted(
            base + timedelta(seconds=self.rng.randint(0, total_seconds))
            for _ in range(expected)
        )
        return times

    def run(self) -> None:
        """Execute the full simulation and print a final report."""
        if self.cfg.verbose:
            print(f"\n{'=' * 60}")
            print(f"  Starting simulation: {self.cfg.dept_name}")
            print(f"  Shift length : {self.cfg.shift_hours}h")
            print(f"  Seed         : {self.cfg.seed}")
            print(f"{'=' * 60}\n")

        arrival_times = self._simulated_timestamps()
        discharge_interval = max(
            1,
            int(self.cfg.arrival_rate_per_hour / self.cfg.discharge_rate_per_hour),
        )

        for i, ts in enumerate(arrival_times):
            # Admit patient
            name, acuity, complaint, attending = random_patient_spec(self.rng)
            try:
                patient = self.dept.admit(
                    name=name,
                    acuity=acuity,
                    chief_complaint=complaint,
                    attending=attending,
                )
                self._admitted_ids.append(patient.patient_id)
                self._log(
                    "arrive",
                    f"{patient.patient_id}  {name:<22}  acuity={acuity}  {complaint}",
                    ts,
                )
            except RuntimeError:
                self._log("arrive", f"WARD FULL — {name} turned away (acuity={acuity})", ts)

            # Periodic discharges
            if i % discharge_interval == 0 and self._admitted_ids:
                pid = self.rng.choice(self._admitted_ids)
                try:
                    p = self.dept.discharge(pid)
                    self._admitted_ids.remove(pid)
                    self._log("discharge", f"{pid}  {p.name}", ts)
                except Exception:
                    pass   # patient may have already been discharged

            # Periodic escalation sweep
            if i % self.cfg.escalation_check_every_n_arrivals == 0:
                n = self.dept.escalate_critical(self.cfg.escalation_threshold)
                if n > 0:
                    self._log(
                        "escalate",
                        f"{n} patient(s) moved to ICU (threshold={self.cfg.escalation_threshold})",
                        ts,
                    )

        # Final snapshot
        if self.cfg.verbose:
            print()
            print(department_report(self.dept))
            print()

        self._log(
            "snapshot",
            f"shift complete — census={self.dept.total_occupancy()}",
            datetime.now(),
        )

    def event_log(self) -> List[Event]:
        return list(self._events)

    def summary(self) -> dict:
        return {
            "total_arrivals":   sum(1 for e in self._events if e.kind == "arrive"),
            "total_discharges": sum(1 for e in self._events if e.kind == "discharge"),
            "total_escalations": sum(1 for e in self._events if e.kind == "escalate"),
            "final_general":    self.dept.general.occupancy(),
            "final_icu":        self.dept.icu.occupancy(),
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description="Run an ER shift simulation.")
    parser.add_argument("--hours",  type=float, default=8.0,  help="Shift length in hours")
    parser.add_argument("--seed",   type=int,   default=42,   help="Random seed")
    parser.add_argument("--rate",   type=float, default=12.0, help="Arrivals per hour")
    parser.add_argument("--quiet",  action="store_true",      help="Suppress per-event output")
    parser.add_argument(
        "--threshold", type=int, default=8,
        help="Acuity threshold for ICU escalation",
    )
    args = parser.parse_args()
    return SimConfig(
        seed=args.seed,
        shift_hours=args.hours,
        arrival_rate_per_hour=args.rate,
        escalation_threshold=args.threshold,
        verbose=not args.quiet,
    )



# ===========================================================================
# SECTION 8: Tests — Patient  (tests/test_patient.py)
# ===========================================================================


"""Tests for the Patient data model."""



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

# ===========================================================================
# SECTION 9: Tests — Registry  (tests/test_registry.py)
# ===========================================================================


"""Tests for PatientRegistry."""



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

# ===========================================================================
# SECTION 10: Tests — Ward  (tests/test_ward.py)
# ===========================================================================


"""Tests for Ward and EmergencyDepartment."""



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

# ===========================================================================
# SECTION 11: Tests — Reports  (tests/test_reports.py)
# ===========================================================================


"""Tests for the reports module."""



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

