"""Filesystem layout shared by the CLI, harness and report.

``TABRECON_DATA_DIR``, ``TABRECON_RESULTS_DIR`` and ``TABRECON_REPORT_DIR`` redirect
the output locations (useful for trial runs that must not touch committed results).
"""
import os
from pathlib import Path

# src/tabrecon/paths.py -> repository root is three levels up
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
DATA_DIR = Path(os.environ.get("TABRECON_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("TABRECON_RESULTS_DIR", ROOT / "results"))
RAW_DIR = RESULTS_DIR / "raw"
REPORT_DIR = Path(os.environ.get("TABRECON_REPORT_DIR", ROOT / "report"))
FIGURES_DIR = REPORT_DIR / "figures"


def dataset_dir(n_rows: int, tag: str = "") -> Path:
    return DATA_DIR / f"rows_{n_rows}{tag}"
