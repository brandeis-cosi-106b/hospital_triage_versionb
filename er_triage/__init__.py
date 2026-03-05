"""ER Triage Registry — public API surface."""

from .patient import Patient, acuity_label, acuity_category
from .registry import PatientRegistry
from .ward import Ward, EmergencyDepartment

__all__ = [
    "Patient",
    "acuity_label",
    "acuity_category",
    "PatientRegistry",
    "Ward",
    "EmergencyDepartment",
]
