"""Discrete-event ER simulation.

Simulates a stream of patient arrivals, acuity assessments, escalations, and
discharges over a configurable time window.  Produces a human-readable log and
a final department report.

Usage
-----
    python -m simulation.simulator                    # default 8-hour shift
    python -m simulation.simulator --hours 4 --seed 99
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

# Ensure the package root is on the path when running as __main__
if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from er_triage import EmergencyDepartment
from er_triage.patient import Patient
from er_triage.reports import department_report
from simulation.data_generator import generate_arrival_batch, random_patient_spec


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


if __name__ == "__main__":
    cfg = _parse_args()
    sim = ERSimulator(cfg)
    sim.run()
    s = sim.summary()
    print(
        f"\nSummary: {s['total_arrivals']} arrived, "
        f"{s['total_discharges']} discharged, "
        f"{s['total_escalations']} escalation sweep(s). "
        f"Final census: {s['final_general']} general / {s['final_icu']} ICU."
    )
