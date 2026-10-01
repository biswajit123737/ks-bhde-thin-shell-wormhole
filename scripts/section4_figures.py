"""
Section 4 figure: Barrow holographic dark energy density rho_BHDE(a) and the
effective shell mass function m(a), for several Barrow exponents Delta.

Generates: bhde_rho_m_vs_a.png

Units: G = c = 1, M = 1. C is expressed via the dimensionless coupling
c_tilde = C / M^(1-Delta), set to 1.0 here for illustration.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

M = 1.0
C_TILDE = 1.0             # dimensionless BHDE coupling (representative value)
DELTA_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)


def rho_bhde(a, Delta, c_tilde=C_TILDE):
    """Barrow HDE density: rho_BHDE(a) = C a^(Delta-2), C = c_tilde * M^(1-Delta)."""
    C = c_tilde * M ** (1 - Delta)
    return C * a ** (Delta - 2)


def m_of_a(a, Delta, c_tilde=C_TILDE):
    """Effective shell mass function: m(a) = 4 pi C a^Delta."""
    C = c_tilde * M ** (1 - Delta)
    return 4 * np.pi * C * a ** Delta


def plot_bhde_rho_m():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    a = np.linspace(0.5, 10, 500)

    for Delta in DELTA_LIST:
        axes[0].loglog(a, rho_bhde(a, Delta), label=f"$\\Delta={Delta}$")
        axes[1].loglog(a, m_of_a(a, Delta), label=f"$\\Delta={Delta}$")

    axes[0].set_xlabel("$a/M$")
    axes[0].set_ylabel(r"$\rho_{BHDE}(a)$")
    axes[0].set_title(r"Barrow HDE density $\rho_{BHDE}=C\,a^{\Delta-2}$")
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("$a/M$")
    axes[1].set_ylabel("$m(a)$")
    axes[1].set_title(r"Effective shell mass $m(a)=4\pi C\,a^{\Delta}$")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "bhde_rho_m_vs_a.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_bhde_rho_m()
    print(f"Figure saved to '{OUTDIR}/bhde_rho_m_vs_a.png'.")
