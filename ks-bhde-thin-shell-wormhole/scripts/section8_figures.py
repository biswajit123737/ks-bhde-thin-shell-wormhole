"""
Section 8 figure: NEC and SEC combinations (sigma+p, sigma+2p) vs throat
radius, for several Barrow exponents Delta.

Generates: energy_conditions.png

Units: G = c = 1, M = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

M = 1.0
Q = 0.3
C_TILDE = 0.3
DELTA_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)


def horizon_radius(q, M=M):
    return np.sqrt(4 * M**2 + q**2)


def NEC(a, Delta, c_tilde=C_TILDE, M=M):
    """sigma + p = C(Delta-2)/2 * a^(Delta-2); strictly negative for Delta in [0,1]."""
    C = c_tilde * M ** (1 - Delta)
    return C * a ** (Delta - 2) * (Delta - 2) / 2


def SEC(a, Delta, c_tilde=C_TILDE, M=M):
    """sigma + 2p = C(Delta-1) * a^(Delta-2); saturates to 0 at Delta=1."""
    C = c_tilde * M ** (1 - Delta)
    return C * a ** (Delta - 2) * (Delta - 1)


def plot_energy_conditions():
    rh = horizon_radius(Q)
    a = np.linspace(rh * 1.02, 8, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for Delta in DELTA_LIST:
        axes[0].plot(a, NEC(a, Delta), label=f"$\\Delta={Delta}$")
        axes[1].plot(a, SEC(a, Delta), label=f"$\\Delta={Delta}$")

    axes[0].axhline(0, color="k", lw=0.7)
    axes[0].set_xlabel("$a/M$")
    axes[0].set_ylabel(r"$\sigma+p$")
    axes[0].set_title("NEC combination (violated for all $\\Delta\\in[0,1]$)")
    axes[0].legend(fontsize=9)

    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set_xlabel("$a/M$")
    axes[1].set_ylabel(r"$\sigma+2p$")
    axes[1].set_title("SEC combination (saturates at $\\Delta=1$)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "energy_conditions.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_energy_conditions()
    print(f"Figure saved to '{OUTDIR}/energy_conditions.png'.")
