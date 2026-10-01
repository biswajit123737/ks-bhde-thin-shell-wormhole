"""
Section 7 figure: (q, Delta) phase diagram of the turning point a*(q,Delta)
and the bounce-then-collapse time tau_c(q,Delta).

Uses a singularity-regularized quadrature (local linearization of V(a) near
the a* endpoint) rather than naive scipy.quad, which is unstable there.

Generates: phase_diagram.png

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
    """Locate the unique turning point via bracketed root-finding, expanding
    the search bracket geometrically until a sign change is found."""
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
    """Integral of 1/sqrt(-V(a)) da from a_lo to a_star. The integrable
    1/sqrt(a*-a) singularity at a_star is handled analytically via local
    linearization, V(a) ~ V'(a*)(a-a*), rather than left to generic
    quadrature (which is numerically unstable there)."""
    Vp_star = Vp(a_star, q, Delta, c_tilde)  # > 0 always, by the monotonicity result
    delta = min(1e-4 * a_star, 0.01 * (a_star - a_lo))
    if delta <= 0 or a_star - delta <= a_lo:
        integrand = lambda a: 1.0 / np.sqrt(max(-V(a, q, Delta, c_tilde), 1e-300))
        val, _ = quad(integrand, a_lo, a_star, limit=400)
        return val
    integrand = lambda a: 1.0 / np.sqrt(max(-V(a, q, Delta, c_tilde), 1e-300))
    bulk, _ = quad(integrand, a_lo, a_star - delta, limit=400)
    tail = 2 * np.sqrt(delta / Vp_star)  # analytic near-singularity contribution
    return bulk + tail


def collapse_time_bounce(a0, a_star, q, Delta, rh, c_tilde=C_TILDE):
    if np.isnan(a_star) or a0 >= a_star:
        return np.nan
    t_up = robust_integral(a0, a_star, q, Delta, c_tilde)
    t_down = robust_integral(rh, a_star, q, Delta, c_tilde)
    return t_up + t_down


def compute_phase_diagram(q_range=(0.01, 1.4), delta_range=(0.0, 0.9), n=26):
    qs = np.linspace(*q_range, n)
    Deltas = np.linspace(*delta_range, n)

    A_star = np.full((len(Deltas), len(qs)), np.nan)
    Tau_c = np.full((len(Deltas), len(qs)), np.nan)

    for i, D in enumerate(Deltas):
        for j, q in enumerate(qs):
            rh = horizon_radius(q)
            if A0 <= rh:
                continue
            astar = find_a_star(q, D)
            if np.isnan(astar) or astar <= A0:
                continue
            A_star[i, j] = astar
            Tau_c[i, j] = collapse_time_bounce(A0, astar, q, D, rh)

    return qs, Deltas, A_star, Tau_c


def plot_phase_diagram():
    qs, Deltas, A_star, Tau_c = compute_phase_diagram()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    im0 = axes[0].pcolormesh(qs, Deltas, np.log10(A_star), shading="auto", cmap="viridis")
    axes[0].set_xlabel("$q/M$")
    axes[0].set_ylabel(r"$\Delta$")
    axes[0].set_title(r"$\log_{10} a^*(q,\Delta)$  (turning radius)")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(qs, Deltas, np.log10(Tau_c), shading="auto", cmap="magma")
    axes[1].set_xlabel("$q/M$")
    axes[1].set_ylabel(r"$\Delta$")
    axes[1].set_title(r"$\log_{10}\tau_c(q,\Delta)$  (bounce-collapse time)")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "phase_diagram.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_phase_diagram()
    print(f"Figure saved to '{OUTDIR}/phase_diagram.png'.")
