"""AISC 360-22 Section F10 — single angles.
Limit states: yielding, lateral-torsional buckling, leg local buckling.

C_b is computed from Eq. F1-1 but taken as not more than 1.5 (F10.2).

Units: kip-inch.
"""

from __future__ import annotations

from math import sqrt
from typing import Optional

from ...typing import FloatLike
from .classification import SlendernessCheck
from .common import (
    AngleBending,
    Axis,
    ElementClass,
    FlexuralStrength,
    LegTip,
    LimitState,
    missing,
    missing_note,
)
from ..constants import YOUNGS_MODULUS_KSI
from .properties import SectionProperties

#: F10.2 caps C_b for single angles.
MAX_CB = 1.5


def calculate_lateral_torsional_buckling_moment(
    elastic_buckling_moment: FloatLike,
    yield_moment: FloatLike,
) -> float:
    """M_n from M_cr and M_y (Eq. F10-2 / F10-3).

    Args:
        elastic_buckling_moment: M_cr
        yield_moment: M_y
    """
    M_cr = float(elastic_buckling_moment)
    M_y = float(yield_moment)
    if M_cr <= 0:
        return 0.0
    if M_y / M_cr <= 1.0:
        return min((1.92 - 1.17 * sqrt(M_y / M_cr)) * M_y, 1.5 * M_y)  # F10-2
    return (0.92 - 0.17 * M_cr / M_y) * M_cr  # F10-3


def calculate_principal_axis_elastic_moment(
    area: FloatLike,
    radius_of_gyration_z: FloatLike,
    leg_thickness: FloatLike,
    unbraced_length: FloatLike,
    angle_asymmetry: FloatLike = 0.0,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """M_cr for major principal axis bending (Eq. F10-4).

    Args:
        area: A_g
        radius_of_gyration_z: r_z, about the minor principal axis
        leg_thickness: t
        unbraced_length: L_b
        angle_asymmetry: beta_w, zero for equal-leg angles; use the negative
            value when the long leg is in compression anywhere along L_b
        ltb_modification_factor: C_b, not more than 1.5
        youngs_modulus: E (ksi)
    """
    A_g = float(area)
    r_z = float(radius_of_gyration_z)
    t = float(leg_thickness)
    L_b = float(unbraced_length)
    beta_w = float(angle_asymmetry)
    C_b = min(float(ltb_modification_factor), MAX_CB)
    E = float(youngs_modulus)
    ratio = beta_w * r_z / (L_b * t)
    return 9 * E * A_g * r_z * t * C_b / (8 * L_b) * (sqrt(1 + 4.4 * ratio**2) + ratio)


def calculate_geometric_axis_elastic_moment(
    leg_width: FloatLike,
    leg_thickness: FloatLike,
    unbraced_length: FloatLike,
    leg_tip: LegTip,
    restrained: bool = False,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """M_cr for geometric-axis bending of an equal-leg angle (Eq. F10-5a / F10-5b).

    Args:
        leg_width: b, width of the leg
        leg_thickness: t
        unbraced_length: L_b
        leg_tip: maximum compression (F10-5a) or tension (F10-5b) at the toe
        restrained: True when laterally restrained at the point of maximum
            moment only; M_cr is then increased 25% per F10.2(b)(2)(ii)
        ltb_modification_factor: C_b, not more than 1.5
        youngs_modulus: E (ksi)
    """
    b = float(leg_width)
    t = float(leg_thickness)
    L_b = float(unbraced_length)
    C_b = min(float(ltb_modification_factor), MAX_CB)
    E = float(youngs_modulus)
    root = sqrt(1 + 0.88 * (L_b * t / b**2) ** 2)
    sign = -1.0 if leg_tip is LegTip.COMPRESSION else 1.0
    M_cr = 0.58 * E * b**4 * t * C_b / L_b**2 * (root + sign)
    return 1.25 * M_cr if restrained else M_cr


def calculate_noncompact_leg_moment(
    yield_stress: FloatLike,
    section_modulus_compression_toe: FloatLike,
    leg_slenderness: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Leg local buckling, noncompact leg (Eq. F10-6)."""
    F_y = float(yield_stress)
    S_c = float(section_modulus_compression_toe)
    lam = float(leg_slenderness)
    return F_y * S_c * (2.43 - 1.72 * lam * sqrt(F_y / float(youngs_modulus)))


def calculate_slender_leg_stress(
    leg_slenderness: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Critical stress for a slender angle leg, F_cr (Eq. F10-8)."""
    return 0.71 * float(youngs_modulus) / float(leg_slenderness) ** 2


def check_f10(
    properties: SectionProperties,
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    leg: Optional[SlendernessCheck],
    bending: AngleBending = AngleBending.PRINCIPAL,
    leg_tip: LegTip = LegTip.COMPRESSION,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F10 check.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        unbraced_length: L_b (in).
        leg: Table B4.1b check for the leg in compression.
        bending: principal-axis or geometric-axis basis.
        leg_tip: whether the leg toe sees maximum compression or tension.
        ltb_modification_factor: C_b (Eq. F1-1), capped internally at 1.5.
        youngs_modulus: E (ksi).
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    L_b = float(unbraced_length)
    C_b = min(float(ltb_modification_factor), MAX_CB)

    result = FlexuralStrength(section_reference="F10", title="Single angles", axis=Axis.MAJOR)
    result.intermediate_values["C_b_used"] = C_b

    geometric = bending is not AngleBending.PRINCIPAL
    if geometric:
        S = properties.section_modulus_x
    else:
        S = properties.section_modulus_principal or properties.section_modulus_x
    if S is None:
        result.add(
            LimitState("Y", "Yielding", "F10-1", note=missing_note(["section_modulus_x"]))
        )
        return result

    # F10.2(b)(2)(i): with no lateral-torsional restraint, M_y and S_c are 0.80
    # of the geometric-axis values.
    unrestrained = bending is AngleBending.GEOMETRIC_UNRESTRAINED
    M_y = F_y * (0.80 * S if unrestrained else S)
    S_c = properties.section_modulus_compression_toe or (0.80 * S if unrestrained else S)
    result.intermediate_values.update({"M_y": M_y, "S": S, "S_c": S_c})
    result.add(LimitState("Y", "Yielding", "F10-1", 1.5 * M_y))

    M_cr = _elastic_moment(properties, bending, leg_tip, L_b, C_b, E)
    if M_cr is None:
        needed = (
            ["area", "radius_of_gyration_z", "leg_thickness"]
            if bending is AngleBending.PRINCIPAL
            else ["leg_width", "leg_thickness"]
        )
        result.add(
            LimitState(
                "LTB", "Lateral-torsional buckling", "F10-4", note=missing_note(needed)
            )
        )
    else:
        result.intermediate_values["M_cr"] = M_cr
        equation = "F10-2" if M_y / M_cr <= 1.0 else "F10-3"
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling",
                equation,
                calculate_lateral_torsional_buckling_moment(M_cr, M_y),
            )
        )

    # --- F10.3 leg local buckling -----------------------------------------
    if leg_tip is not LegTip.COMPRESSION:
        result.add(
            LimitState(
                "LLB",
                "Leg local buckling",
                "F10-6",
                note="leg toe in tension — the limit state does not apply",
            )
        )
    elif leg is None or not leg.known:
        result.add(LimitState("LLB", "Leg local buckling", "F10-6", note=missing_note(["b/t"])))
    elif leg.classification is ElementClass.COMPACT:
        result.add(
            LimitState(
                "LLB",
                "Leg local buckling",
                "F10-6",
                note="compact leg — the limit state does not apply",
            )
        )
    elif leg.classification is ElementClass.NONCOMPACT:
        result.add(
            LimitState(
                "LLB",
                "Leg local buckling (noncompact)",
                "F10-6",
                calculate_noncompact_leg_moment(F_y, S_c, leg.slenderness, E),
            )
        )
    else:
        F_cr = calculate_slender_leg_stress(leg.slenderness, E)
        result.intermediate_values["F_cr_leg"] = F_cr
        result.add(LimitState("LLB", "Leg local buckling (slender)", "F10-7 / F10-8", F_cr * S_c))
    return result


def _elastic_moment(
    properties: SectionProperties,
    bending: AngleBending,
    leg_tip: LegTip,
    unbraced_length: float,
    ltb_modification_factor: float,
    youngs_modulus: float,
) -> Optional[float]:
    """M_cr per F10.2, or None when the required properties are absent."""
    if unbraced_length <= 0:
        return None
    if bending is AngleBending.PRINCIPAL:
        if missing(properties, ["area", "radius_of_gyration_z", "leg_thickness"]):
            return None
        return calculate_principal_axis_elastic_moment(
            properties.area,
            properties.radius_of_gyration_z,
            properties.leg_thickness,
            unbraced_length,
            properties.angle_asymmetry or 0.0,
            ltb_modification_factor,
            youngs_modulus,
        )
    if missing(properties, ["leg_width", "leg_thickness"]):
        return None
    return calculate_geometric_axis_elastic_moment(
        properties.leg_width,
        properties.leg_thickness,
        unbraced_length,
        leg_tip,
        bending is AngleBending.GEOMETRIC_RESTRAINED,
        ltb_modification_factor,
        youngs_modulus,
    )
