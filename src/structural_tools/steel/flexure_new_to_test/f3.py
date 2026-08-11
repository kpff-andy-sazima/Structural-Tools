"""AISC 360-22 Section F3 — doubly symmetric I-shapes with compact webs and
noncompact or slender flanges, major axis. Limit states: LTB (per F2.2) and
compression flange local buckling.

Units: kip-inch.
"""

from __future__ import annotations

from ...typing import FloatLike
from .classification import SlendernessCheck
from .common import (
    ElementClass,
    FlexuralStrength,
    LimitState,
    interpolate_noncompact,
    missing_note,
)
from ..constants import YOUNGS_MODULUS_KSI
from .f2 import calculate_plastic_moment, check_f2
from .properties import SectionProperties


def calculate_noncompact_flange_moment(
    plastic_moment: FloatLike,
    yield_stress: FloatLike,
    section_modulus_x_axis: FloatLike,
    flange_slenderness: FloatLike,
    limiting_compact: FloatLike,
    limiting_noncompact: FloatLike,
) -> float:
    """Compression flange local buckling, noncompact flange (Eq. F3-1)."""
    return interpolate_noncompact(
        float(plastic_moment),
        0.7 * float(yield_stress) * float(section_modulus_x_axis),
        flange_slenderness,
        limiting_compact,
        limiting_noncompact,
    )


def calculate_slender_flange_moment(
    flange_buckling_coefficient: FloatLike,
    section_modulus_x_axis: FloatLike,
    flange_slenderness: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Compression flange local buckling, slender flange (Eq. F3-2).

    Args:
        flange_buckling_coefficient: k_c, 0.35 <= k_c <= 0.76
        section_modulus_x_axis: S_x
        flange_slenderness: lambda = b_f/2t_f
        youngs_modulus: E (ksi)
    """
    k_c = float(flange_buckling_coefficient)
    S_x = float(section_modulus_x_axis)
    lam = float(flange_slenderness)
    E = float(youngs_modulus)
    return 0.9 * E * k_c * S_x / lam**2


def check_f3(
    properties: SectionProperties,
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    flange: SlendernessCheck | None,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F3 check: F2.2 lateral-torsional buckling plus flange local buckling.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        unbraced_length: L_b (in).
        flange: Table B4.1b check for the compression flange.
        ltb_modification_factor: C_b (Eq. F1-1).
        youngs_modulus: E (ksi).
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)

    result = check_f2(properties, F_y, unbraced_length, ltb_modification_factor, E)
    result.section_reference = "F3"
    result.title = (
        "Doubly symmetric I-shaped members with compact webs and noncompact or "
        "slender flanges bent about their major axis"
    )
    # F3 does not include a yielding limit state; F2.2 LTB is capped at M_p anyway.
    result.limit_states = [ls for ls in result.limit_states if ls.key != "Y"]

    if flange is None or not flange.known:
        result.add(
            LimitState(
                "FLB",
                "Compression flange local buckling",
                "F3-1 / F3-2",
                note=missing_note(["flange_slenderness"]),
            )
        )
        return result

    if flange.classification is ElementClass.COMPACT:
        result.add(
            LimitState(
                "FLB",
                "Compression flange local buckling",
                "F3-1",
                note="compact flange — the limit state does not apply",
            )
        )
        return result

    M_p = calculate_plastic_moment(F_y, properties.plastic_modulus_x)
    S_x = properties.section_modulus_x
    if flange.classification is ElementClass.NONCOMPACT:
        M_n = calculate_noncompact_flange_moment(
            M_p, F_y, S_x, flange.slenderness, flange.limiting_compact, flange.limiting_noncompact
        )
        result.add(
            LimitState("FLB", "Compression flange local buckling (noncompact)", "F3-1", M_n)
        )
    else:
        k_c = properties.flange_buckling_coefficient or 0.76
        result.intermediate_values["k_c"] = k_c
        M_n = calculate_slender_flange_moment(k_c, S_x, flange.slenderness, E)
        result.add(LimitState("FLB", "Compression flange local buckling (slender)", "F3-2", M_n))
    return result
