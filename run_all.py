"""
Regenerate every figure of the paper.

Run from the repository root:

    python run_all.py

All figures are written to ./figs. Total run time is about one minute
on a laptop.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "section2_figures.py",             # Figs. 1-2
    "schwarzschild_vs_ks_figure.py",   # Fig. 3
    "section3_figures.py",             # Fig. 4
    "section4_figures.py",             # Fig. 5
    "section5_figures.py",             # Fig. 6
    "force_decomposition_figure.py",   # Fig. 7
    "section6_figures.py",             # Fig. 8
    "section7_figures.py",             # Figs. 9 and 11
    "phase_space_figure.py",           # Fig. 10
    "phase_diagram_figure.py",         # Fig. 12
    "collapse_time_contour_figure.py", # Fig. 13
    "critical_curve_figure.py",        # Fig. 14
    "scaling_law_figure.py",           # Fig. 15
    "section8_figures.py",             # Fig. 16
]

failed = []
for name in SCRIPTS:
    t0 = time.time()
    print(f"--> {name}")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / name)], cwd=ROOT)
    if r.returncode != 0:
        failed.append(name)
    print(f"    done in {time.time() - t0:.1f} s\n")

if failed:
    sys.exit("Failed: " + ", ".join(failed))
print("All figures written to", ROOT / "figs")
