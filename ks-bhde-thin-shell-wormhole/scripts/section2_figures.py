"""
Section 2 figures: geometry of the Kazakov-Solodukhin quantum-corrected black hole.

Generates:
  1. f_r_various_q.png          -- metric function f(r) for several q
  2. horizon_vs_q.png           -- horizon radius r_h vs deformation q
  3. ricci_scalar_divergence.png    -- R(r) diverging as r -> q+
  4. kretschmann_divergence.png     -- K(r) diverging as r -> q+

Units: G = c = 1, and M = 1 (all lengths measured in units of the mass).
Edit the CONFIG block below to change parameter values, ranges, or styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------- CONFIG -----------------------------
M = 1.0                      # black hole mass (sets the unit system)
OUTDIR = "figs"               # output directory for the PNGs
DPI = 150

Q_LIST_FOR_F = [0.0, 0.3, 0.6, 0.9]   # deformation values to overlay in fig 1
Q_FOR_CURVATURE_PLOTS = 0.5           # single q used for figs 3 and 4
Q_RANGE_FOR_HORIZON = (0.0, 2.0)      # q range for fig 2
# --------------------------------------------------------------------

os.makedirs(OUTDIR, exist_ok=True)


# ----------------------- Core metric functions -----------------------
def f(r, q, M=M):
    """KS metric function f(r) = sqrt(r^2-q^2)/r - 2M/r."""
    return np.sqrt(np.clip(r**2 - q**2, 0, None)) / r - 2 * M / r


def fprime(r, q, M=M):
    """f'(r) = 2M/r^2 + q^2 / (r^2 sqrt(r^2-q^2))."""
    return 2 * M / r**2 + q**2 / (r**2 * np.sqrt(np.clip(r**2 - q**2, 1e-12, None)))


def fdoubleprime(r, q, M=M):
    """f''(r), derived and verified symbolically (see Section 2 notes)."""
    rq = np.sqrt(np.clip(r**2 - q**2, 1e-12, None))
    return -4 * M / r**3 - q**2 / (r * rq**3) - 2 / (r * rq) + 2 * rq / r**3


def horizon_radius(q, M=M):
    """r_h = sqrt(4M^2 + q^2)."""
    return np.sqrt(4 * M**2 + q**2)


def ricci_scalar(r, q):
    """R(r) for the KS metric (matches the literature formula exactly)."""
    term1 = (2 / r**2) * (1 - (1 - q**2 / r**2) ** (-0.5))
    term2 = (q**2 / r**4) * (1 - q**2 / r**2) ** (-1.5)
    return term1 + term2


def kretschmann_scalar(r, q, M=M):
    """K(r) = f''^2 + 4 f'^2 / r^2 + 4 (1-f)^2 / r^4."""
    fp = fprime(r, q, M)
    fpp = fdoubleprime(r, q, M)
    fr = f(r, q, M)
    return fpp**2 + 4 * fp**2 / r**2 + 4 * (1 - fr) ** 2 / r**4


# ----------------------------- Figure 1 -----------------------------
def plot_f_r_various_q():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = np.linspace(0.05, 6, 2000)
    for q in Q_LIST_FOR_F:
        mask = r > q
        ax.plot(r[mask], f(r[mask], q), label=f"$q={q}$")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("$r/M$")
    ax.set_ylabel("$f(r)$")
    ax.set_title("KS metric function for various deformation $q$ ($M=1$)")
    ax.legend()
    ax.set_ylim(-2, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "f_r_various_q.png"), dpi=DPI)
    plt.close()


# ----------------------------- Figure 2 -----------------------------
def plot_horizon_vs_q():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    qvals = np.linspace(*Q_RANGE_FOR_HORIZON, 300)
    rh = horizon_radius(qvals)
    ax.plot(qvals, rh)
    ax.set_xlabel("$q/M$")
    ax.set_ylabel("$r_h/M$")
    ax.set_title(r"Horizon radius $r_h=\sqrt{4M^2+q^2}$")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "horizon_vs_q.png"), dpi=DPI)
    plt.close()


# ----------------------------- Figure 3 -----------------------------
def plot_ricci_divergence():
    q = Q_FOR_CURVATURE_PLOTS
    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = np.linspace(q * 1.001, q * 3, 3000)
    ax.plot(r, ricci_scalar(r, q))
    ax.axvline(q, color="r", ls="--", label="$r=q$")
    ax.set_xlabel("$r/M$")
    ax.set_ylabel("$R(r)$")
    ax.set_title(f"Ricci scalar diverging as $r\\to q^+$  ($q={q}$)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "ricci_scalar_divergence.png"), dpi=DPI)
    plt.close()


# ----------------------------- Figure 4 -----------------------------
def plot_kretschmann_divergence():
    q = Q_FOR_CURVATURE_PLOTS
    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = np.linspace(q * 1.001, q * 3, 3000)
    ax.plot(r, kretschmann_scalar(r, q))
    ax.axvline(q, color="r", ls="--", label="$r=q$")
    ax.set_xlabel("$r/M$")
    ax.set_ylabel("$K(r)$")
    ax.set_title(f"Kretschmann scalar diverging as $r\\to q^+$  ($q={q}$)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "kretschmann_divergence.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_f_r_various_q()
    plot_horizon_vs_q()
    plot_ricci_divergence()
    plot_kretschmann_divergence()
    print(f"All 4 figures saved to '{OUTDIR}/'.")
