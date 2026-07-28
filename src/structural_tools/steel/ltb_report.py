"""Static, self-contained HTML report for AISC 360-22 F2 lateral-torsional
buckling. Renders an M_n vs. L_b diagram with matplotlib (Agg) and embeds it
as a base64 PNG in a standalone HTML string. No web server; no extra deps.
"""

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .section import WSection
from .flexure import calculate_nominal_flexural_strength


def _figure_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_ltb_report(
    section: WSection,
    yield_stress: float,
    *,
    resistance_factor: float = 0.90,
    ltb_modification_factor: float = 1.0,
    design_unbraced_length: float | None = None,
    n_points: int = 400,
) -> str:
    """Return a self-contained HTML string for an LTB check.

    Lengths in inches; moments plotted in kip-ft. design_unbraced_length, if
    given, marks a point on the curve (also in inches).
    """

    def mn(lb_in: float):
        return calculate_nominal_flexural_strength(section, yield_stress, lb_in, ltb_modification_factor)

    base = mn(1e-6)  # L_p, L_r, M_p are independent of L_b
    L_p, L_r, M_p = base.limiting_yield_length, base.limiting_inelastic_length, base.plastic_moment

    lb = np.linspace(1e-6, 1.6 * L_r, n_points)
    mn_vals = np.array([mn(x).nominal_moment for x in lb])

    lb_ft, mn_kft = lb / 12.0, mn_vals / 12.0
    phi_mn = resistance_factor * mn_kft
    label = section.name or "section"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(lb_ft, mn_kft, lw=2.2, label=r"$M_n$ (nominal)")
    ax.plot(lb_ft, phi_mn, lw=1.8, ls="--", label=rf"$\phi M_n$  ($\phi={resistance_factor}$)")
    ax.axhline(M_p / 12.0, color="grey", ls=":", lw=1, label=r"$M_p$")
    ax.axvline(L_p / 12.0, color="green", ls=":", lw=1)
    ax.axvline(L_r / 12.0, color="red", ls=":", lw=1)
    ax.annotate(r"$L_p$", (L_p / 12.0, 0), textcoords="offset points", xytext=(3, 6), color="green")
    ax.annotate(r"$L_r$", (L_r / 12.0, 0), textcoords="offset points", xytext=(3, 6), color="red")

    if design_unbraced_length is not None:
        res = mn(design_unbraced_length)
        ax.scatter(
            [design_unbraced_length / 12.0],
            [res.nominal_moment / 12.0],
            zorder=5,
            s=55,
            color="black",
            label=f"design $L_b$ ({res.region})",
        )

    ax.set_xlabel(r"Unbraced length  $L_b$  (ft)")
    ax.set_ylabel(r"Flexural strength  (kip-ft)")
    ax.set_title(f"AISC 360-22 F2 — LTB capacity: {label}")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    img = _figure_to_base64(fig)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>LTB Report — {label}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1b1b1b; }}
  table {{ border-collapse: collapse; margin-top: 1rem; }}
  td, th {{ border: 1px solid #ccc; padding: 4px 12px; text-align: right; }}
  th {{ background: #f3f3f3; }}
  img {{ max-width: 100%; height: auto; margin-top: 1rem; }}
</style></head><body>
<h1>Lateral-Torsional Buckling — {label}</h1>
<p>AISC 360-22 Section F2 (compact, major-axis bending). F_y = {yield_stress} ksi.</p>
<table>
  <tr><th>Quantity</th><th>Value</th><th>Unit</th></tr>
  <tr><td>M_p</td><td>{M_p / 12.0:,.1f}</td><td>kip-ft</td></tr>
  <tr><td>L_p</td><td>{L_p / 12.0:,.2f}</td><td>ft</td></tr>
  <tr><td>L_r</td><td>{L_r / 12.0:,.2f}</td><td>ft</td></tr>
</table>
data:image/png;base64,{img}
</body></html>"""
