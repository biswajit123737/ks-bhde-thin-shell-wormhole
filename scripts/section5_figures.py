"""
Section 5 figure: preview of the effective potential V(a), showing how the
Barrow exponent Delta and the quantum deformation q each reshape it.

Generates: V_a_preview.png

Units: G = c = 1, M = 1. C is expressed via the dimensionless coupling
c_tilde = C / M^(1-Delta).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

M = 1.0
C_TILDE = 0.15   # representative BHDE coupling, increased for clearer Delta-separation
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)


def f(r, q, M=M):
    return np.sqrt(np.clip(r**2 - q**2, 0, None)) / r - 2 * M / r


def horizon_radius(q, M=M):
    return np.sqrt(4 * M**2 + q**2)


def V(a, q, Delta, c_tilde=C_TILDE, M=M):
    """Closed-form effective potential, Eq. (V-closed-form)."""
    C = c_tilde * M ** (1 - Delta)
    return f(a, q, M) - 4 * np.pi**2 * C**2 * a ** (2 * Delta - 2)


def plot_V_preview():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: vary Delta at fixed q
    q_fixed = 0.3
    a = np.linspace(horizon_radius(q_fixed) * 1.02, 8, 800)
    for Delta in [0.0, 0.3, 0.6, 0.9]:
        axes[0].plot(a, V(a, q_fixed, Delta), label=f"$\\Delta={Delta}$")
    axes[0].axhline(0, color="k", lw=0.7)
    axes[0].set_xlabel("$a/M$")
    axes[0].set_ylabel("$V(a)$")
    axes[0].set_title(f"Effect of $\\Delta$ on $V(a)$  ($q={q_fixed}$)")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-1.0, 1.2)

    # Right: vary q at fixed Delta
    Delta_fixed = 0.5
    for q in [0.0, 0.3, 0.6, 0.9]:
        a = np.linspace(horizon_radius(q) * 1.02, 8, 800)
        axes[1].plot(a, V(a, q, Delta_fixed), label=f"$q={q}$")
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set_xlabel("$a/M$")
    axes[1].set_ylabel("$V(a)$")
    axes[1].set_title(f"Effect of $q$ on $V(a)$  ($\\Delta={Delta_fixed}$)")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(-1.0, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "V_a_preview.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_V_preview()
    print(f"Figure saved to '{OUTDIR}/V_a_preview.png'.")
