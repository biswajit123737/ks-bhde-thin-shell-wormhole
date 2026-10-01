"""
Section 6 figure: effective force decomposition F(a) = F_geom(a) + F_BHDE(a).

Generates: force_decomposition.png

Units: G = c = 1, M = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

M = 1.0
Q, C_TILDE = 0.3, 0.3
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)


def horizon_radius(q, M=M):
    return np.sqrt(4 * M**2 + q**2)


def F_geom(a, q, M=M):
    """Geometric contribution: -f'(a)/2, independent of matter content."""
    return -M / a**2 - q**2 / (2 * a**2 * np.sqrt(a**2 - q**2))


def F_bhde(a, Delta, c_tilde=C_TILDE, M=M):
    """BHDE contribution: sourced entirely by the Barrow-HDE shell."""
    C = c_tilde * M ** (1 - Delta)
    return 4 * np.pi**2 * C**2 * a ** (2 * Delta - 3) * (Delta - 1)


def plot_force_decomposition():
    rh = horizon_radius(Q)
    a = np.linspace(rh * 1.02, 10, 500)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for Delta in [0.0, 0.5, 0.9]:
        axes[0].plot(a, F_bhde(a, Delta), label=f"$F_{{BHDE}}$, $\\Delta={Delta}$")
    axes[0].plot(a, F_geom(a, Q), "k--", lw=2, label=r"$F_{geom}$ (independent of $\Delta$)")
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_xlabel("$a/M$")
    axes[0].set_ylabel("Force")
    axes[0].set_title("Force contributions (both always negative: inward)")
    axes[0].legend(fontsize=8)

    for Delta in [0.0, 0.5, 0.9]:
        Ftot = F_geom(a, Q) + F_bhde(a, Delta)
        axes[1].plot(a, Ftot, label=f"$\\Delta={Delta}$")
    axes[1].axhline(0, color="gray", lw=0.6)
    axes[1].set_xlabel("$a/M$")
    axes[1].set_ylabel(r"$F(a)=F_{geom}+F_{BHDE}$")
    axes[1].set_title("Total force: never positive, no balance point")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "force_decomposition.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_force_decomposition()
    print(f"Figure saved to '{OUTDIR}/force_decomposition.png'.")
