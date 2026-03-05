"""Synthetic patient and scenario data generation.

Produces realistic-looking (but entirely fictional) patient records for use
in simulation runs and stress testing.
"""

from __future__ import annotations

import random
from typing import List, Tuple

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
