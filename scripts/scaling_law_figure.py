"""
Section 7.9 figure: scaling of the bounce-then-collapse time tau_c and of
the turning point a* as Delta -> 1^-.

Both are plotted as ln(.) against 1/(1 - Delta); a straight line means
exponential-in-1/(1-Delta) growth. The script also prints the power-law
diagnostics quoted in the text (effective local exponent and the best
power-law fit), so the comparison can be reproduced.

tau_c is computed by quadrature,
    tau_c = int_{a0}^{a*} da / sqrt(-V)  +  int_{r_h}^{a*} da / sqrt(-V),
with the substitution a = a* - u^2 removing the integrable 1/sqrt
singularity at the turning point. A direct ODE integration is run for
two values of Delta as an independent cross-check.

Parameters: q = 0.3, c = 0.3, a0 = 2.5, initially expanding throat.

Generates: scaling_law.png

Units: G = c = 1, M = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.stats import linregress
import os

M = 1.0
Q, C_TILDE, A0 = 0.3, 0.3, 2.5
DELTAS = np.array([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93])
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)
R_H = np.sqrt(4 * M**2 + Q**2)


def f(a, q=Q):
    return np.sqrt(a**2 - q**2) / a - 2 * M / a


def V(a, D, c=C_TILDE):
    return f(a) - 4 * np.pi**2 * c**2 * a**(2 * D - 2)


def Vp(a, D, c=C_TILDE, q=Q):
    return (2 * M / a**2 + q**2 / (a**2 * np.sqrt(a**2 - q**2))
            + 8 * np.pi**2 * c**2 * (1 - D) * a**(2 * D - 3))


def turning_point(D):
    hi = 10.0
    while V(hi, D) < 0:
        hi *= 2
    return brentq(V, R_H * (1 + 1e-12), hi, args=(D,), xtol=1e-13, rtol=1e-14)


def leg(lo, a_star, D):
    """int_lo^{a*} da / sqrt(-V(a)), regularised at a*."""
    def g(u):
        if u == 0:
            return 2 / np.sqrt(Vp(a_star, D))
        return 2 * u / np.sqrt(-V(a_star - u * u, D))
    return quad(g, 0, np.sqrt(a_star - lo), limit=400, epsabs=1e-12, epsrel=1e-12)[0]


def tau_bounce(D):
    a_star = turning_point(D)
    return leg(A0, a_star, D) + leg(R_H, a_star, D), a_star


def tau_ode(D):
    def horizon(t, y):
        return y[0] - R_H
    horizon.terminal, horizon.direction = True, -1
    sol = solve_ivp(lambda t, y: [y[1], -0.5 * Vp(y[0], D)], [0, 1e5],
                    [A0, np.sqrt(-V(A0, D))], events=horizon,
                    rtol=1e-10, atol=1e-12, max_step=5)
    return sol.t_events[0][0]


tau, a_st = np.array([tau_bounce(D) for D in DELTAS]).T
for D, t, a in zip(DELTAS, tau, a_st):
    print(f"Delta = {D:.2f}:  a* = {a:10.5g}   tau_c = {t:10.6g}")
for D in (0.5, 0.8):
    print(f"ODE cross-check, Delta = {D}: tau_c = {tau_ode(D):.6g}")

x = 1 / (1 - DELTAS)
fit_t = linregress(x, np.log(tau))
fit_a = linregress(x, np.log(a_st))
print(f"ln tau_c = {fit_t.slope:.4f}/(1-Delta) + {fit_t.intercept:.4f},  R^2 = {fit_t.rvalue**2:.5f}")
print(f"ln a*    = {fit_a.slope:.4f}/(1-Delta) + {fit_a.intercept:.4f},  R^2 = {fit_a.rvalue**2:.5f}")

# power-law diagnostics (Section 7.9)
lx, ly = np.log(1 - DELTAS), np.log(tau)
print("effective local exponent alpha_eff:", np.round(-np.diff(ly) / np.diff(lx), 2))
pw = linregress(lx, ly)
print(f"best power-law fit: alpha = {-pw.slope:.3f}, R^2 = {pw.rvalue**2:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
xx = np.linspace(x.min(), x.max(), 100)
panels = [(axes[0], np.log(tau), fit_t, "C0", r"\ln\tau_c", "Collapse time"),
          (axes[1], np.log(a_st), fit_a, "C2", r"\ln a^*", "Turning point")]
for ax, y, fit, col, lab, name in panels:
    ax.plot(x, y, "o", ms=8, color=col, label="numerical data")
    ax.plot(xx, fit.intercept + fit.slope * xx, "r--",
            label=rf"fit: ${lab}={fit.slope:.3f}/(1-\Delta)+{fit.intercept:.2f}$" + "\n"
                  + rf"$R^2={fit.rvalue**2:.4f}$")
    ax.set_xlabel(r"$1/(1-\Delta)$")
    ax.set_ylabel(rf"${lab}$")
    ax.set_title(f"{name}: exponential divergence")
    ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "scaling_law.png"), dpi=DPI)
print("saved", os.path.join(OUTDIR, "scaling_law.png"))
