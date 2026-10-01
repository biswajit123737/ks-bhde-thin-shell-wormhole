"""
Section 7.3 figure: phase portrait of the throat dynamics in the (a, adot)
plane, with the direction of flow on each branch.

Every admissible trajectory lies on the constraint curve adot^2 = -V(a).
The upper branch (adot > 0) is the expanding phase, the lower branch
(adot < 0) the contracting phase; they meet at the turning point a* and
end at the horizon r_h.

Generates: phase_space_family.png

Units: G = c = 1, M = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os

M = 1.0
Q, DELTA, C_TILDE = 0.3, 0.5, 0.3   # representative parameter choice
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)


def f(a, q, M=M):
    return np.sqrt(a**2 - q**2) / a - 2 * M / a


def V(a, q=Q, delta=DELTA, c=C_TILDE):
    return f(a, q) - 4 * np.pi**2 * c**2 * a**(2 * delta - 2)


r_h = np.sqrt(4 * M**2 + Q**2)
a_star = brentq(V, r_h * (1 + 1e-9), 1e3, xtol=1e-12, rtol=1e-12)
print(f"r_h = {r_h:.6f},  a* = {a_star:.6f}")

a = np.linspace(r_h, a_star, 2000)
adot = np.sqrt(np.clip(-V(a), 0, None))

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.axvspan(r_h, a_star, color="#2e7d32", alpha=0.07)
ax.axhline(0, color="grey", lw=0.6)
ax.axvline(r_h, color="blue", ls=":", lw=1.8, label=rf"$r_h={r_h:.2f}$ (collapse)")
ax.axvline(a_star, color="red", ls="--", lw=1.5, label=rf"$a^*={a_star:.2f}$ (turning point)")
ax.plot(np.r_[a, a[::-1]], np.r_[adot, -adot[::-1]], color="C0", lw=2.2,
        label=r"constraint $\dot a^2=-V(a)$")

# flow arrows: expansion (a increasing) on the upper branch,
# contraction (a decreasing) on the lower branch
def arrow(x0, sign, color):
    y0 = sign * np.sqrt(-V(x0))
    dx = 0.02 * sign
    x1 = x0 + dx
    y1 = sign * np.sqrt(max(-V(x1), 0.0))
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=22))

for x0 in np.linspace(r_h + 0.45, a_star - 0.5, 4):
    arrow(x0, +1, "darkblue")
    arrow(x0 + 0.15, -1, "darkred")

ax.text(4.3, 0.55, "bounce-then-collapse\n(upper $\\to$ turning $\\to$ lower)", color="darkblue")
ax.text(3.3, -0.85, "direct collapse", color="darkred")
ax.set_xlim(r_h - 0.2, a_star + 0.85)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r"$a/M$")
ax.set_ylabel(r"$\dot a$")
ax.set_title(rf"Phase portrait with flow direction ($q={Q},\Delta={DELTA},\tilde c={C_TILDE}$)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "phase_space_family.png"), dpi=DPI)
print("saved", os.path.join(OUTDIR, "phase_space_family.png"))
