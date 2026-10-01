"""
Section 3 figure: static-case surface energy density sigma(a) and pressure p(a)
implied by the Darmois-Israel junction conditions alone (no matter model yet).

Generates: sigma_p_static.png

Units: G = c = 1, M = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

M = 1.0
OUTDIR = "figs"
DPI = 150
Q_LIST = [0.0, 0.3, 0.6, 0.9]   # deformation values to compare

os.makedirs(OUTDIR, exist_ok=True)


def f(r, q, M=M):
    return np.sqrt(np.clip(r**2 - q**2, 0, None)) / r - 2 * M / r


def fprime(r, q, M=M):
    return 2 * M / r**2 + q**2 / (r**2 * np.sqrt(np.clip(r**2 - q**2, 1e-12, None)))


def horizon_radius(q, M=M):
    return np.sqrt(4 * M**2 + q**2)


def sigma_static(a, q):
    """Static-case (adot=addot=0) surface energy density, Eq. (sigma) with f only."""
    fa = f(a, q)
    return -np.sqrt(np.clip(fa, 0, None)) / (2 * np.pi * a)


def p_static(a, q):
    """Static-case surface pressure, Eq. (pressure) with adot=addot=0."""
    fa = f(a, q)
    fpa = fprime(a, q)
    return (2 * fa + a * fpa) / (8 * np.pi * a * np.sqrt(np.clip(fa, 1e-12, None)))


def plot_sigma_p_static():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for q in Q_LIST:
        rh = horizon_radius(q)
        a = np.linspace(rh * 1.01, rh * 4, 500)  # throat must sit outside the horizon
        axes[0].plot(a, sigma_static(a, q), label=f"$q={q}$")
        axes[1].plot(a, p_static(a, q), label=f"$q={q}$")

    axes[0].set_xlabel("$a/M$")
    axes[0].set_ylabel(r"$\sigma(a)$")
    axes[0].set_title("Static surface energy density (always $<0$: exotic matter)")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].legend()

    axes[1].set_xlabel("$a/M$")
    axes[1].set_ylabel("$p(a)$")
    axes[1].set_title("Static surface pressure")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "sigma_p_static.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_sigma_p_static()
    print(f"Figure saved to '{OUTDIR}/sigma_p_static.png'.")
