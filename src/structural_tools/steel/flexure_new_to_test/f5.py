"""AISC 360-22 Section F5 — doubly and singly symmetric I-shaped members with
slender webs, major axis.

Limit states: compression flange yielding, lateral-torsional buckling,
compression flange local buckling, tension flange yielding.

Units: kip-inch.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Optional

from ...typing import FloatLike
from .classification import SlendernessCheck
from .common import (
    Axis,
    ElementClass,
    FlexuralStrength,
    LimitState,
    missing,
    missing_note,
)
from ..constants import YOUNGS_MODULUS_KSI
from .f4 import calculate_limiting_yield_length
from .properties import SectionProperties


def calculate_bending_strength_reduction_factor(
    web_area_ratio: FloatLike,
    web_slenderness: FloatLike,
    yield_stress: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Bending strength reduction factor R_pg (Eq. F5-6).

    Args:
        web_area_ratio: a_w per Eq. F4-12, not to exceed 10
        web_slenderness: h_c/t_w
        yield_stress: F_y (ksi)
        youngs_modulus: E (ksi)
    """
    a_w = min(float(web_area_ratio), 10.0)
    slenderness = float(web_slenderness)
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    R_pg = 1.0 - a_w / (1200 + 300 * a_w) * (slenderness - 5.7 * sqrt(E / F_y))
    return min(R_pg, 1.0)


def calculate_limiting_inelastic_length(
    yield_stress: FloatLike,
    compression_flange_radius: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Limiting unbraced length for inelastic LTB, L_r (Eq. F5-5)."""
    return pi * float(compression_flange_radius) * sqrt(
        float(youngs_modulus) / (0.7 * float(yield_stress))
    )


def check_f5(
    properties: SectionProperties,
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    flange: Optional[SlendernessCheck],
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F5 check.

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
    C_b = float(ltb_modification_factor)
    L_b = float(unbraced_length)

    result = FlexuralStrength(
        section_reference="F5",
        title=(
            "Doubly symmetric and singly symmetric I-shaped members with slender "
            "webs bent about their major axis"
        ),
        axis=Axis.MAJOR,
    )
    absent = missing(
        properties,
        [
            "section_modulus_compression",
            "section_modulus_tension",
            "compression_flange_radius",
            "web_area_ratio",
            "web_slenderness",
        ],
    )
    if absent:
        result.add(
            LimitState("CFY", "Compression flange yielding", "F5-1", note=missing_note(absent))
        )
        return result

    S_xc = properties.section_modulus_compression
    S_xt = properties.section_modulus_tension
    r_t = properties.compression_flange_radius

    R_pg = calculate_bending_strength_reduction_factor(
        properties.web_area_ratio, properties.web_slenderness, F_y, E
    )
    L_p = calculate_limiting_yield_length(F_y, r_t, E)  # Eq. F4-7
    L_r = calculate_limiting_inelastic_length(F_y, r_t, E)  # Eq. F5-5
    result.intermediate_values.update(
        {
            "R_pg": R_pg,
            "a_w": properties.web_area_ratio,
            "L_p": L_p,
            "L_r": L_r,
            "r_t": r_t,
            "M_yc": F_y * S_xc,
        }
    )

    result.add(LimitState("CFY", "Compression flange yielding", "F5-1", R_pg * F_y * S_xc))

    if L_b <= L_p:
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling",
                "F5-2",
                note="L_b <= L_p — the limit state does not apply",
            )
        )
    elif L_b <= L_r:
        F_cr = min(C_b * (F_y - 0.3 * F_y * (L_b - L_p) / (L_r - L_p)), F_y)  # F5-3
        result.add(
            LimitState(
                "LTB", "Lateral-torsional buckling (inelastic)", "F5-2 / F5-3", R_pg * F_cr * S_xc
            )
        )
    else:
        F_cr = min(C_b * pi**2 * E / (L_b / r_t) ** 2, F_y)  # F5-4
        result.add(
            LimitState(
                "LTB", "Lateral-torsional buckling (elastic)", "F5-2 / F5-4", R_pg * F_cr * S_xc
            )
        )

    if flange is None or not flange.known:
        result.add(
            LimitState(
                "FLB",
                "Compression flange local buckling",
                "F5-7",
                note=missing_note(["flange_slenderness"]),
            )
        )
    elif flange.classification is ElementClass.COMPACT:
        result.add(
            LimitState(
                "FLB",
                "Compression flange local buckling",
                "F5-7",
                note="compact flange — the limit state does not apply",
            )
        )
    elif flange.classification is ElementClass.NONCOMPACT:
        F_cr = F_y - 0.3 * F_y * (flange.slenderness - flange.limiting_compact) / (
            flange.limiting_noncompact - flange.limiting_compact
        )  # F5-8
        result.add(
            LimitState(
                "FLB",
                "Compression flange local buckling (noncompact)",
                "F5-7 / F5-8",
                R_pg * F_cr * S_xc,
            )
        )
    else:
        k_c = properties.flange_buckling_coefficient or 0.76
        F_cr = 0.9 * E * k_c / flange.slenderness**2  # F5-9
        result.add(
            LimitState(
                "FLB",
                "Compression flange local buckling (slender)",
                "F5-7 / F5-9",
                R_pg * F_cr * S_xc,
            )
        )

    if S_xt >= S_xc:
        result.add(
            LimitState(
                "TFY",
                "Tension flange yielding",
                "F5-10",
                note="S_xt >= S_xc — the limit state does not apply",
            )
        )
    else:
        result.add(LimitState("TFY", "Tension flange yielding", "F5-10", F_y * S_xt))
    return result
