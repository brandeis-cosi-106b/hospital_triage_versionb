# ER Triage Registry

A data-driven emergency room triage management system written in Python.

## Overview

The ER Triage Registry tracks patients across an emergency department, supporting
fast admission, discharge, and ward-level operations based on clinical acuity scores.

Acuity is scored on a 1–10 scale where **10 = immediately life-threatening** and
**1 = non-urgent**. The system supports efficient range queries (e.g., "how many
patients currently have acuity ≥ 8?") and threshold-based escalation (e.g.,
"transfer all critical patients to the ICU").

## Structure

```
er_triage/          Core library
  patient.py        Patient data model
  _index.py         Internal sorted index (private)
  registry.py       PatientRegistry — primary public API
  ward.py           Ward and EmergencyDepartment models
  reports.py        Census and load reporting

simulation/         Standalone ER simulation harness
  data_generator.py Random patient / scenario generation
  simulator.py      Discrete-event simulation loop

tests/              Pytest test suite
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                         # run test suite
python -m simulation.simulator # run a sample simulation
```

## Usage

```python
from er_triage import PatientRegistry, EmergencyDepartment

dept = EmergencyDepartment("County General ER")

dept.admit("Maria Santos",  acuity=9, chief_complaint="chest pain")
dept.admit("Jerome Okafor", acuity=3, chief_complaint="sprained ankle")
dept.admit("Lin Wei",       acuity=7, chief_complaint="severe abdominal pain")

print(dept.general.registry.most_acute())   # most severe patient
dept.escalate_critical(threshold=8)         # move acuity >= 8 to ICU
print(dept.census())
```

## Acuity Scale

| Score | Label            | Category |
|-------|------------------|----------|
| 1–3   | Low–Moderate     | Green    |
| 4–6   | Urgent–High      | Yellow   |
| 7     | Severe           | Orange   |
| 8–10  | Critical–Immediate | Red    |
