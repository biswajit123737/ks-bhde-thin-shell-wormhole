"""
Section 7.8 figure: critical curve Delta_crit(q), the Barrow exponent at
which the turning point a* reaches a prescribed multiple of the horizon
radius, a_target = k * r_h(q).

Setting V(a_target) = 0 and solving for Delta gives the closed form
(Eq. 38 of the paper)

    Delta_crit = 1 + ln[ f(a_target) / (4 pi^2 c^2) ] / (2 ln a_target).

Generates: critical_curve.png

Units: G = c = 1, M = 1.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

M = 1.0
C_TILDE = 0.3
REACH = [1.5, 2.0, 3.0, 5.0, 10.0]   # a_target / r_h
OUTDIR = "figs"
DPI = 150

os.makedirs(OUTDIR, exist_ok=True)


def f(a, q, M=M):
    return np.sqrt(a**2 - q**2) / a - 2 * M / a


def delta_crit(q, a_target, c=C_TILDE):
    return 1 + np.log(f(a_target, q) / (4 * np.pi**2 * c**2)) / (2 * np.log(a_target))


# consistency check against the value quoted in Section 7.8:
# q = 0.3, c = 0.3, a_target = a* = 5.5577  ->  Delta_crit = 0.4997
print("check: Delta_crit(q=0.3, a_target=5.5577) =", round(delta_crit(0.3, 5.5577), 4))

q = np.linspace(0.01, 1.4, 400)
r_h = np.sqrt(4 * M**2 + q**2)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.axhspan(0, 1, color="#2e7d32", alpha=0.07)
ax.axhline(1, color="grey", ls=":", lw=1)
for k in REACH:
    ax.plot(q, delta_crit(q, k * r_h), lw=1.8, label=rf"$a^*/r_h={k}$")
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"$q/M$")
ax.set_ylabel(r"$\Delta_{\rm crit}(q)$")
ax.set_title(r"Critical curve: $\Delta$ required for $a^*$ to reach a given multiple of $r_h$")
ax.grid(alpha=0.3)
ax.legend(title="target reach", loc="lower center")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "critical_curve.png"), dpi=DPI)
print("saved", os.path.join(OUTDIR, "critical_curve.png"))
