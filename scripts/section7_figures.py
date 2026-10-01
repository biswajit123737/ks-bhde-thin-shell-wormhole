"""
Section 7 figures: numerical integration of the throat equation of motion,
classifying trajectories into direct-collapse and bounce-then-collapse cases,
and mapping collapse time across parameter space.

Generates:
  trajectories.png     -- a(tau) and phase portrait for both cases
  collapse_time.png    -- collapse time vs Delta and vs q

Units: G = c = 1, M = 1.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os

M = 1.0
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)


def f(a, q, M=M):
    return np.sqrt(a**2 - q**2) / a - 2 * M / a


def fp(a, q, M=M):
    return 2 * M / a**2 + q**2 / (a**2 * np.sqrt(a**2 - q**2))


def V(a, q, Delta, C):
    return f(a, q) - 4 * np.pi**2 * C**2 * a ** (2 * Delta - 2)


def Vp(a, q, Delta, C):
    return fp(a, q) + 8 * np.pi**2 * C**2 * (1 - Delta) * a ** (2 * Delta - 3)


def horizon_radius(q, M=M):
    return np.sqrt(4 * M**2 + q**2)


def rhs(tau, y, q, Delta, C):
    a, adot = y
    return [adot, -0.5 * Vp(a, q, Delta, C)]


def make_event(q):
    def event_horizon(tau, y, q, Delta, C):
        return y[0] - horizon_radius(q) * 1.0001
    event_horizon.terminal = True
    event_horizon.direction = -1
    return event_horizon


def collapse_time(q, Delta, c_tilde, a0, sign=+1, tmax=500):
    """Integrate to horizon crossing; returns proper time, or NaN if unreached/invalid."""
    C = c_tilde * M ** (1 - Delta)
    V0 = V(a0, q, Delta, C)
    if V0 >= 0:
        return np.nan  # a0 outside the dynamically accessible range for these parameters
    adot0 = sign * np.sqrt(-V0)
    ev = make_event(q)
    sol = solve_ivp(rhs, [0, tmax], [a0, adot0], args=(q, Delta, C), events=ev,
                     max_step=0.05, rtol=1e-9, atol=1e-11)
    if len(sol.t_events[0]) == 0:
        return np.nan
    return sol.t_events[0][0]


def plot_trajectories():
    q, Delta, c_tilde = 0.3, 0.5, 0.3
    C = c_tilde * M ** (1 - Delta)
    rh = horizon_radius(q)
    a0 = 4.0
    V0 = V(a0, q, Delta, C)
    adot0_mag = np.sqrt(-V0)
    ev = make_event(q)

    solA = solve_ivp(rhs, [0, 20], [a0, -adot0_mag], args=(q, Delta, C), events=ev,
                      dense_output=True, max_step=0.02, rtol=1e-10, atol=1e-12)
    solB = solve_ivp(rhs, [0, 20], [a0, +adot0_mag], args=(q, Delta, C), events=ev,
                      dense_output=True, max_step=0.02, rtol=1e-10, atol=1e-12)

    # constraint residual check
    for name, sol in [("A", solA), ("B", solB)]:
        a_t, adot_t = sol.y
        residual = np.max(np.abs(adot_t**2 + V(a_t, q, Delta, C)))
        print(f"Case {name}: max constraint residual = {residual:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(solA.t, solA.y[0], label="direct collapse ($\\dot a(0)<0$)", lw=2)
    axes[0].plot(solB.t, solB.y[0], label="bounce then collapse ($\\dot a(0)>0$)", lw=2)
    axes[0].axhline(rh, color="k", ls=":", lw=1, label="$r_h$")
    axes[0].set_xlabel(r"$\tau/M$")
    axes[0].set_ylabel("$a/M$")
    axes[0].set_title(f"Throat radius vs. proper time ($q={q},\\Delta={Delta},\\tilde c={c_tilde}$)")
    axes[0].legend(fontsize=9)

    a_range = np.linspace(rh * 1.001, 6.2, 500)
    adot_upper = np.sqrt(np.clip(-V(a_range, q, Delta, C), 0, None))
    axes[1].plot(a_range, adot_upper, "gray", lw=1, ls="--", label=r"constraint $\dot a^2=-V(a)$")
    axes[1].plot(a_range, -adot_upper, "gray", lw=1, ls="--")
    axes[1].plot(solA.y[0], solA.y[1], lw=2, label="direct collapse")
    axes[1].plot(solB.y[0], solB.y[1], lw=2, label="bounce then collapse")
    axes[1].axvline(rh, color="k", ls=":", lw=1)
    axes[1].set_xlabel("$a/M$")
    axes[1].set_ylabel(r"$\dot a$")
    axes[1].set_title("Phase portrait")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "trajectories.png"), dpi=DPI)
    plt.close()


def plot_collapse_time():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    c_tilde, a0 = 0.3, 2.5  # close to horizon: dynamically valid across a wide parameter range

    q_fixed = 0.3
    Deltas = np.linspace(0.0, 0.97, 20)
    tc_direct = [collapse_time(q_fixed, D, c_tilde, a0, sign=-1) for D in Deltas]
    tc_bounce = [collapse_time(q_fixed, D, c_tilde, a0, sign=+1) for D in Deltas]
    axes[0].semilogy(Deltas, tc_direct, "o-", label="direct collapse")
    axes[0].semilogy(Deltas, tc_bounce, "s-", label="bounce then collapse")
    axes[0].set_xlabel(r"$\Delta$")
    axes[0].set_ylabel(r"$\tau_c/M$ (log scale)")
    axes[0].set_title(f"Collapse time vs $\\Delta$  ($q={q_fixed}$)")
    axes[0].legend(fontsize=9)

    Delta_fixed = 0.5
    qs = np.linspace(0.0, 1.5, 20)
    tc_direct_q = [collapse_time(qv, Delta_fixed, c_tilde, a0, sign=-1) for qv in qs]
    tc_bounce_q = [collapse_time(qv, Delta_fixed, c_tilde, a0, sign=+1) for qv in qs]
    axes[1].plot(qs, tc_direct_q, "o-", label="direct collapse")
    axes[1].plot(qs, tc_bounce_q, "s-", label="bounce then collapse")
    axes[1].set_xlabel("$q$")
    axes[1].set_ylabel(r"$\tau_c/M$")
    axes[1].set_title(f"Collapse time vs $q$  ($\\Delta={Delta_fixed}$)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "collapse_time.png"), dpi=DPI)
    plt.close()


if __name__ == "__main__":
    plot_trajectories()
    plot_collapse_time()
    print(f"Figures saved to '{OUTDIR}/'.")
