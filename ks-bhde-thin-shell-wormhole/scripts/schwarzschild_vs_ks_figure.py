"""
Section 2 figure: Schwarzschild vs KS curvature invariants, side by side,
to isolate the effect of the quantum correction directly.

Generates: schwarzschild_vs_KS_curvature.png

Units: G = c = 1, M = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

M = 1.0
OUTDIR = "figs"
DPI = 150
Q_VAL = 0.5   # KS deformation parameter used for the comparison

os.makedirs(OUTDIR, exist_ok=True)


def f(r, q, M=M):
    return np.sqrt(np.clip(r**2 - q**2, 0, None)) / r - 2 * M / r


def fprime(r, q, M=M):
    return 2 * M / r**2 + q**2 / (r**2 * np.sqrt(np.clip(r**2 - q**2, 1e-12, None)))


def fdoubleprime(r, q, M=M):
    rq = np.sqrt(np.clip(r**2 - q**2, 1e-12, None))
    return -4 * M / r**3 - q**2 / (r * rq**3) - 2 / (r * rq) + 2 * rq / r**3


def ricci_scalar(r, q):
    """R(r); identically 0 for Schwarzschild (q=0), the vacuum case."""
    if q == 0:
        return np.zeros_like(r)
    return (2 / r**2) * (1 - (1 - q**2 / r**2) ** (-0.5)) + (q**2 / r**4) * (
        1 - q**2 / r**2
    ) ** (-1.5)


def kretschmann_scalar(r, q, M=M):
    """K(r); reduces to the textbook 48 M^2 / r^6 for Schwarzschild (q=0)."""
    if q == 0:
        return 48 * M**2 / r**6
    fp = fprime(r, q, M)
    fpp = fdoubleprime(r, q, M)
    fr = f(r, q, M)
    return fpp**2 + 4 * fp**2 / r**2 + 4 * (1 - fr) ** 2 / r**4


def plot_schwarzschild_vs_ks():
    LW, VLW = 2.6, 1.3
    FS_TITLE, FS_LABEL, FS_LEGEND, FS_TICK = 18, 16, 13, 13

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left: Kretschmann, log-log ---
    r_schw = np.linspace(0.01, 6, 4000)
    r_ks = np.linspace(Q_VAL * 1.0005, 6, 4000)

    axes[0].loglog(r_schw, kretschmann_scalar(r_schw, 0.0), label="Schwarzschild", lw=LW)
    axes[0].loglog(r_ks, kretschmann_scalar(r_ks, Q_VAL), label=f"KS ($q={Q_VAL}$)", lw=LW)
    axes[0].axvline(Q_VAL, color="gray", ls=":", lw=VLW, label="$r=q$")
    axes[0].set_xlabel("$r/M$", fontsize=FS_LABEL)
    axes[0].set_ylabel("$K(r)$", fontsize=FS_LABEL)
    axes[0].set_title("Kretschmann curvature invariant", fontsize=FS_TITLE)
    axes[0].legend(fontsize=FS_LEGEND)
    axes[0].tick_params(labelsize=FS_TICK)

    # --- Right: Ricci scalar, pure log scale (R>0 everywhere for KS, r>q) ---
    r_lin = np.linspace(Q_VAL * 1.0005, 6, 4000)
    axes[1].loglog(r_lin, ricci_scalar(r_lin, Q_VAL), label=f"KS ($q={Q_VAL}$): $R\\neq 0$", lw=LW, color="C1")
    axes[1].axvline(Q_VAL, color="gray", ls=":", lw=VLW, label="$r=q$")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-6, 1e5)
    axes[1].set_xlabel("$r/M$", fontsize=FS_LABEL)
    axes[1].set_ylabel("$R(r)$", fontsize=FS_LABEL)
    axes[1].set_title("Ricci curvature scalar", fontsize=FS_TITLE)
    # Schwarzschild's R=0 cannot be represented on a log axis; noted via annotation instead
    axes[1].text(
        0.97, 0.06, "Schwarzschild: $R\\equiv 0$ (not shown, log scale)",
        transform=axes[1].transAxes, ha="right", fontsize=11, style="italic", color="C0",
    )
    axes[1].legend(fontsize=FS_LEGEND, loc="upper right")
    axes[1].tick_params(labelsize=FS_TICK)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "schwarzschild_vs_KS_curvature.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_schwarzschild_vs_ks()
    print(f"Figure saved to '{OUTDIR}/schwarzschild_vs_KS_curvature.png'.")
