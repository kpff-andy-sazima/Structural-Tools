"""Modeling tools"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from Pynite import FEModel3D

from ..steel.constants import YOUNGS_MODULUS_KSI
from ..steel.flexure import Bracing, evaluate_beam_flexure
from ..typing import FloatLike
from .postprocessing import get_moments_from_tags

if TYPE_CHECKING:
    from ..steel.section import WSection


def analyze_member_flexure(
    model: FEModel3D,
    section: WSection,
    member_name: str = "m_1",
    tags: str | list[str] = "strength",
    top_flange_brace_points: Sequence[FloatLike] | Bracing = Bracing.CONTINUOUS,
    bottom_flange_brace_points: Sequence[FloatLike] | Bracing = Bracing.UNBRACED,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
    positive_moment_compresses: str = "bottom",
    n_points: int = 500,
    moment_scale: float = 1 / 12,
    phi_b: float = 0.9,
    zero_atol=1e-6,
) -> pd.DataFrame:
    """Tidy one-row-per-segment DataFrame indexed by load combination.

    For each combo and each unbraced segment, record geometry, L_b, C_b,
    limit state, phi*M_n, M_u, and the demand/capacity ratio. Moments are
    scaled by `moment_scale` (default kip-in -> kip-ft).
    Numeric values within `zero_atol` of zero are snapped to exactly 0.0.
    Args:
        results_dict: mapping combo label -> list of FlexuralSegmentResult
            (exactly what your zip(labels, moments) loop builds).
        phi_b: flexural resistance factor applied to M_n (e.g. 0.90).
        moment_scale: multiplier applied to every moment column; default
            1/12 converts kip-in to kip-ft. Use 1.0 to keep kip-in.
        zero_atol: absolute tolerance below which numeric cells are set to 0.
    Returns:
        pandas.DataFrame indexed by "Load Combo", sorted by Load Combo then x_start.
    """
    x, moments, labels, _ = get_moments_from_tags(model=model, member_name=member_name, tags=tags, n_points=n_points)
    fy = model.members[member_name].material.fy
    if fy is None:
        raise ValueError(f"Material '{model.members[member_name].material.fy}' has no fy")

    results_dict = {}
    for combo_name, moment_list in zip(labels, moments):
        results_dict[combo_name] = evaluate_beam_flexure(
            section=section,
            yield_stress=fy,
            positions=x,
            moments=moment_list,
            top_flange_brace_points=top_flange_brace_points,
            bottom_flange_brace_points=bottom_flange_brace_points,
            youngs_modulus=youngs_modulus,
            positive_moment_compresses=positive_moment_compresses,
        )

    rows = []
    for combo, results in results_dict.items():
        for r in results:
            ltb = r.lateral_torsional_buckling
            phi_Mn = phi_b * ltb.nominal_moment
            rows.append({
                "Load Combo": combo,
                "flange": r.flange,
                "segment start": r.x_start,
                "segment end": r.x_end,
                "unbraced length": ltb.unbraced_length,
                "ltb modification factor": r.ltb_modification_factor,
                "limit region": ltb.region,
                "moment demand": r.moment_demand * moment_scale,
                "factored moment capacity": phi_Mn * moment_scale,
                "DCR": r.moment_demand / phi_Mn,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["Load Combo", "segment start"])

    # snap near-zero floats to exactly 0.0 (avoids -0.0 and 1e-16 noise)
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(np.isclose(df[num], 0.0, atol=zero_atol), 0.0)

    return df.set_index("Load Combo")
