# Dynamical thin-shell wormholes: Kazakov–Solodukhin + Barrow holographic dark energy

Numerical codes for the paper

> B. Sarkar and U. Debnath, *Dynamical Thin-Shell Wormholes from the Kazakov–Solodukhin Quantum-Corrected Black Hole Supported by Barrow Holographic Dark Energy* (submitted).

Every figure in the paper can be regenerated from this repository, and the numbers quoted in the text (turning points, collapse times, scaling-law fits) are printed by the scripts that produce them.

## Quick start

```bash
git clone https://github.com/<your-username>/ks-bhde-thin-shell-wormhole.git
cd ks-bhde-thin-shell-wormhole
pip install -r requirements.txt
python run_all.py
```

The figures are written to `figs/`. The full run takes about a minute on a laptop. Each script can also be run on its own from the repository root, e.g. `python scripts/scaling_law_figure.py`.

## What produces what

| Paper figure | Script | Output file |
|---|---|---|
| Figs. 1–2 | `section2_figures.py` | `f_r_various_q.png`, `horizon_vs_q.png` |
| Fig. 3 | `schwarzschild_vs_ks_figure.py` | `schwarzschild_vs_KS_curvature.png` |
| Fig. 4 | `section3_figures.py` | `sigma_p_static.png` |
| Fig. 5 | `section4_figures.py` | `bhde_rho_m_vs_a.png` |
| Fig. 6 | `section5_figures.py` | `V_a_preview.png` |
| Fig. 7 | `force_decomposition_figure.py` | `force_decomposition.png` |
| Fig. 8 | `section6_figures.py` | `V_monotonicity.png` |
| Figs. 9, 11 | `section7_figures.py` | `trajectories.png`, `collapse_time.png` |
| Fig. 10 | `phase_space_figure.py` | `phase_space_family.png` |
| Fig. 12 | `phase_diagram_figure.py` | `phase_diagram.png` |
| Fig. 13 | `collapse_time_contour_figure.py` | `collapse_time_contour.png` |
| Fig. 14 | `critical_curve_figure.py` | `critical_curve.png` |
| Fig. 15 | `scaling_law_figure.py` | `scaling_law.png` |
| Fig. 16 | `section8_figures.py` | `energy_conditions.png` |

`section2_figures.py` also writes two single-invariant plots (`ricci_scalar_divergence.png`, `kretschmann_divergence.png`) that are not used in the paper; the combined comparison of Fig. 3 replaces them.

## Model and conventions

- Geometrized units, G = c = 1, and M = 1 throughout, so every length (q, a, r_h) is in units of M.
- KS metric function: f(r) = √(r² − q²)/r − 2M/r, horizon r_h = √(4M² + q²).
- Effective potential of the throat: V(a) = f(a) − 4π²C² a^(2Δ−2), with equation of motion ȧ² + V(a) = 0.
- The dimensionless coupling is written `C_TILDE` (c̃ in the paper). Representative values used in most figures: q = 0.3, Δ = 0.5, c̃ = 0.3.

## Numerical methods

- Trajectories: `scipy.integrate.solve_ivp` (explicit Runge–Kutta 4(5), Dormand–Prince), relative tolerance 10⁻⁹–10⁻¹⁰, absolute tolerance 10⁻¹¹–10⁻¹², with a terminal event at the horizon.
- Turning points and the critical curve: Brent's method (`scipy.optimize.brentq`) applied to V(a) = 0.
- Collapse times: adaptive quadrature (`scipy.integrate.quad`) of ∫ da/√(−V). In `scaling_law_figure.py` the substitution a = a* − u² removes the integrable singularity at the turning point, and a direct ODE integration is run as a cross-check.

## Checks you can reproduce

Running the scripts prints, among other things:

- a* = 5.5612 and r_h = 2.0224 for q = 0.3, Δ = 0.5, c̃ = 0.3 (`phase_space_figure.py`);
- Δ_crit = 0.5000 when the critical-curve formula is evaluated at that a* (`critical_curve_figure.py`);
- ln τ_c = 0.694/(1 − Δ) + 1.43 with R² = 0.9997, and ln a* = 0.597/(1 − Δ) + 0.43 with R² = 0.9992, together with the effective power-law exponent rising from 1.8 to 8.1 and the best power-law fit (R² ≈ 0.94) that the paper rejects (`scaling_law_figure.py`).

## Requirements

Python 3 with NumPy, SciPy and Matplotlib (see `requirements.txt`). Tested with Python 3.11, NumPy 2.4.4, SciPy 1.17.1 and Matplotlib 3.10.9.

## Citation

If you use this code, please cite the paper above. Citation metadata for the software itself is in `CITATION.cff`.

## License

MIT; see `LICENSE`.

## Contact

Biswajit Sarkar, Department of Mathematics, IIEST Shibpur — biswajitsarkar17091996@gmail.com
