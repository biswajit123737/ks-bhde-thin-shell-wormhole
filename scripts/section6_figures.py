"""
Section 6 figure: effective potential V(a) and its derivative V'(a) for a
representative parameter choice, illustrating strict monotonicity of V'(a)
and the resulting single turning point a*.

Generates: V_monotonicity.png

Units: G = c = 1, M = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

M = 1.0
Q, DELTA, C_TILDE = 0.3, 0.5, 0.3   # representative parameter choice
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)


def f(a, q, M=M):
    return np.sqrt(np.clip(a**2 - q**2, 0, None)) / a - 2 * M / a


def fp(a, q, M=M):
    return 2 * M / a**2 + q**2 / (a**2 * np.sqrt(np.clip(a**2 - q**2, 1e-12, None)))


def horizon_radius(q, M=M):
    return np.sqrt(4 * M**2 + q**2)


def V(a, q=Q, Delta=DELTA, c_tilde=C_TILDE, M=M):
    C = c_tilde * M ** (1 - Delta)
    return f(a, q, M) - 4 * np.pi**2 * C**2 * a ** (2 * Delta - 2)


def Vp(a, q=Q, Delta=DELTA, c_tilde=C_TILDE, M=M):
    """V'(a) in the term-by-term-positive form used in the monotonicity proof."""
    C = c_tilde * M ** (1 - Delta)
    return fp(a, q, M) + 8 * np.pi**2 * C**2 * (1 - Delta) * a ** (2 * Delta - 3)


def plot_V_monotonicity():
    rh = horizon_radius(Q)
    a = np.linspace(rh * 1.001, rh * 8, 2000)

    Vvals = V(a)
    Vpvals = Vp(a)

    # locate the single turning point a* where V(a) crosses zero
    idx = np.where(np.diff(np.sign(Vvals)))[0][0]
    a_star = a[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(a, Vvals, lw=2.2)
    axes[0].axhline(0, color="k", lw=0.7, label="$V=0$")
    axes[0].axvline(rh, color="b", ls=":", lw=1.4, label=f"$r_h={rh:.2f}$")
    axes[0].axvline(a_star, color="r", ls="--", lw=1.2, label=f"$a^*={a_star:.2f}$")
    axes[0].fill_between(a, Vvals, 0, where=(Vvals < 0), color="green", alpha=0.15,
                          label="dynamically allowed")
    axes[0].set_xlabel("$a/M$")
    axes[0].set_ylabel("$V(a)$")
    axes[0].set_title(f"Effective potential ($q={Q},\\Delta={DELTA},\\tilde c={C_TILDE}$)")
    axes[0].legend(fontsize=8)

    axes[1].plot(a, Vpvals, lw=2.2, color="C1")
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].axvline(rh, color="b", ls=":", lw=1.4)
    axes[1].axvline(a_star, color="r", ls="--", lw=1.2)
    axes[1].set_xlabel("$a/M$")
    axes[1].set_ylabel("$V'(a)$")
    axes[1].set_title("Derivative: strictly positive everywhere")
    axes[1].set_ylim(0, Vpvals.max() * 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "V_monotonicity.png"), dpi=DPI)
    plt.close()

    return a_star, rh


if __name__ == "__main__":
    a_star, rh = plot_V_monotonicity()
    print(f"Figure saved to '{OUTDIR}/V_monotonicity.png'. Turning point a* = {a_star:.4f}, r_h = {rh:.4f}")
