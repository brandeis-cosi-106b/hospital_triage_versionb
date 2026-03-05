"""Build onelargefile.py from the package sources."""
import re, os

HEADER = '''\
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

'''

SECTION_TEMPLATE = """\
# ===========================================================================
# SECTION {n}: {title}
# ===========================================================================

"""

# Single-line import patterns to drop
DROP_LINE_PATTERNS = [
    re.compile(r'^\s*from __future__ import'),
    re.compile(r'^\s*from \.(patient|_store|registry|ward|reports) import'),
    re.compile(r'^\s*from er_triage(\.[a-z_]+)? import'),
    re.compile(r'^\s*from simulation\.[a-z_]+ import'),
    re.compile(r'^\s*import (argparse|random|re|sys|uuid|pytest)\b'),
    re.compile(r'^\s*from dataclasses import'),
    re.compile(r'^\s*from datetime import'),
    re.compile(r'^\s*from typing import'),
    re.compile(r'^\s*if TYPE_CHECKING:'),
    re.compile(r'^\s*import os\s*$'),
]

SECTIONS = [
    ("Patient Data Model",          "er_triage/patient.py"),
    ("Internal Patient Store",      "er_triage/_store.py"),
    ("Patient Registry",            "er_triage/registry.py"),
    ("Ward and Emergency Dept",     "er_triage/ward.py"),
    ("Reports and Metrics",         "er_triage/reports.py"),
    ("Simulation — Data Generator", "simulation/data_generator.py"),
    ("Simulation — Simulator",      "simulation/simulator.py"),
    ("Tests — Patient",             "tests/test_patient.py"),
    ("Tests — Registry",            "tests/test_registry.py"),
    ("Tests — Ward",                "tests/test_ward.py"),
    ("Tests — Reports",             "tests/test_reports.py"),
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

out_lines = [HEADER]

for idx, (title, filepath) in enumerate(SECTIONS, start=1):
    out_lines.append(SECTION_TEMPLATE.format(n=idx, title=f"{title}  ({filepath})"))
    with open(filepath) as f:
        src = f.read()

    skip_main_block = False   # for if __name__ == '__main__' bodies
    skip_import_parens = False  # for multi-line parenthesised imports being dropped

    for line in src.splitlines():
        stripped = line.strip()

        # ---- Handle multi-line parenthesised imports that were opened ----
        if skip_import_parens:
            if ')' in line:
                skip_import_parens = False
            continue

        # ---- Handle __main__ guard blocks ----
        if re.match(r'^if __name__\s*==\s*["\']__main__["\']', stripped):
            skip_main_block = True
            continue
        if skip_main_block:
            if line and not line[0].isspace():
                skip_main_block = False
            else:
                continue

        # ---- Check single-line drop patterns ----
        drop = any(p.match(line) for p in DROP_LINE_PATTERNS)
        if drop:
            # If the import opens a parenthesis that isn't closed, skip following lines
            if '(' in line and ')' not in line:
                skip_import_parens = True
            continue

        out_lines.append(line)

    out_lines.append("")

result = "\n".join(out_lines) + "\n"
with open("onelargefile.py", "w") as f:
    f.write(result)

print(f"Written {len(result.splitlines())} lines")
