"""
Section 7 figure + analysis: labeled contour map of the dimensionless
collapse time tau_c(q, Delta), plus local logarithmic sensitivity
analysis S_q = d(ln tau_c)/d(ln q) and S_Delta = d(ln tau_c)/d(Delta).

Generates: collapse_time_contour.png

Units: G = c = 1, M = 1.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os

M = 1.0
C_TILDE, A0 = 0.3, 2.5
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)


def f(a, q, M=M):
    return np.sqrt(a**2 - q**2) / a - 2 * M / a


def fp(a, q, M=M):
    return 2 * M / a**2 + q**2 / (a**2 * np.sqrt(a**2 - q**2))


def horizon_radius(q, M=M):
    return np.sqrt(4 * M**2 + q**2)


def V(a, q, Delta, c_tilde=C_TILDE, M=M):
    C = c_tilde * M ** (1 - Delta)
    return f(a, q, M) - 4 * np.pi**2 * C**2 * a ** (2 * Delta - 2)


def Vp(a, q, Delta, c_tilde=C_TILDE, M=M):
    C = c_tilde * M ** (1 - Delta)
    return fp(a, q, M) + 8 * np.pi**2 * C**2 * (1 - Delta) * a ** (2 * Delta - 3)


def find_a_star(q, Delta, c_tilde=C_TILDE, a_hi_start=10.0, max_tries=80):
    rh = horizon_radius(q)
    a_lo = rh * 1.0001
    a_hi = a_hi_start
    tries = 0
    while V(a_hi, q, Delta, c_tilde) < 0:
        a_hi *= 10
        tries += 1
        if tries > max_tries:
            return np.nan
    return brentq(lambda a: V(a, q, Delta, c_tilde), a_lo, a_hi, xtol=1e-8, rtol=1e-12)


def robust_integral(a_lo, a_star, q, Delta, c_tilde=C_TILDE):
    """Handles the integrable 1/sqrt(a*-a) endpoint singularity analytically
    via local linearization of V(a) near a_star, avoiding quad instability."""
    Vp_star = Vp(a_star, q, Delta, c_tilde)
    delta = min(1e-4 * a_star, 0.01 * (a_star - a_lo))
    if delta <= 0 or a_star - delta <= a_lo:
        integrand = lambda a: 1.0 / np.sqrt(max(-V(a, q, Delta, c_tilde), 1e-300))
        val, _ = quad(integrand, a_lo, a_star, limit=400)
        return val
    integrand = lambda a: 1.0 / np.sqrt(max(-V(a, q, Delta, c_tilde), 1e-300))
    bulk, _ = quad(integrand, a_lo, a_star - delta, limit=400)
    tail = 2 * np.sqrt(delta / Vp_star)
    return bulk + tail


def tau_c(q, Delta, c_tilde=C_TILDE, a0=A0):
    rh = horizon_radius(q)
    astar = find_a_star(q, Delta, c_tilde)
    if np.isnan(astar) or astar <= a0:
        return np.nan
    return robust_integral(a0, astar, q, Delta, c_tilde) + robust_integral(rh, astar, q, Delta, c_tilde)


def local_sensitivities(q0, D0, h_q=0.02, h_D=0.01):
    """Central-difference logarithmic sensitivities of tau_c to q and Delta."""
    S_q = (np.log(tau_c(q0 + h_q, D0)) - np.log(tau_c(q0 - h_q, D0))) / (2 * h_q) * q0
    S_D = (np.log(tau_c(q0, D0 + h_D)) - np.log(tau_c(q0, D0 - h_D))) / (2 * h_D)
    return S_q, S_D


def plot_contour():
    qs = np.linspace(0.01, 1.4, 60)
    Deltas = np.linspace(0.0, 0.9, 60)
    Q, D = np.meshgrid(qs, Deltas)
    T = np.full_like(Q, np.nan)
    for i in range(len(Deltas)):
        for j in range(len(qs)):
            T[i, j] = tau_c(qs[j], Deltas[i])

    logT = np.log10(T)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    levels = np.arange(np.floor(np.nanmin(logT)), np.ceil(np.nanmax(logT)) + 0.5, 0.5)
    cf = ax.contourf(Q, D, logT, levels=levels, cmap="magma")
    cs = ax.contour(Q, D, logT, levels=levels[::2], colors="white", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt=lambda v: f"$10^{{{v:.1f}}}$")
    ax.set_xlabel("$q/M$")
    ax.set_ylabel(r"$\Delta$")
    ax.set_title(r"Contour map of dimensionless collapse time $\tau_c(q,\Delta)/M$")
    fig.colorbar(cf, ax=ax, label=r"$\log_{10}\tau_c$")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "collapse_time_contour.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_contour()
    print(f"Figure saved to '{OUTDIR}/collapse_time_contour.png'.")

    print("\nLocal sensitivity analysis:")
    for q0, D0 in [(0.2, 0.3), (0.8, 0.3), (0.2, 0.7), (0.8, 0.7)]:
        Sq, SD = local_sensitivities(q0, D0)
        print(f"  (q,Delta)=({q0},{D0}): S_q={Sq:.4f}, S_Delta={SD:.4f}, |S_Delta/S_q|={abs(SD/Sq):.1f}")
