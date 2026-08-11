"""AISC 360-22 Section F7 — square and rectangular HSS and box sections, either
axis. Limit states: yielding, flange local buckling, web local buckling,
lateral-torsional buckling.

360-22 replaced the 360-16 closed-form F7-2/F7-6 expressions (the 3.57/4.0 and
0.305/0.738 coefficients) with the standard lambda interpolation; that is what
is implemented here.

Units: kip-inch.
"""

from __future__ import annotations

from math import sqrt
from typing import Optional

from ...typing import FloatLike
from .classification import SlendernessCheck
from .common import (
    Axis,
    ElementClass,
    FlexuralStrength,
    LimitState,
    interpolate_noncompact,
    missing,
    missing_note,
)
from ..constants import YOUNGS_MODULUS_KSI
from .f5 import calculate_bending_strength_reduction_factor
from .properties import SectionProperties


def calculate_effective_width(
    flange_slenderness: FloatLike,
    wall_thickness: FloatLike,
    yield_stress: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
    box_section: bool = False,
) -> float:
    """Effective width b_e of a slender compression flange (Eq. F7-4 / F7-5).

    Args:
        flange_slenderness: b/t
        wall_thickness: t (design wall thickness)
        yield_stress: F_y (ksi)
        youngs_modulus: E (ksi)
        box_section: True for box sections (Eq. F7-5, 0.34), False for HSS
            (Eq. F7-4, 0.38)
    """
    lam = float(flange_slenderness)
    t = float(wall_thickness)
    root = sqrt(float(youngs_modulus) / float(yield_stress))
    coefficient = 0.34 if box_section else 0.38
    b_e = 1.92 * t * root * (1 - coefficient / lam * root)
    return min(b_e, lam * t)


def calculate_effective_section_modulus(
    properties: SectionProperties,
    axis: Axis,
    yield_stress: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> Optional[float]:
    """S_e for a slender-flange HSS or box section (Eq. F7-3).

    The compression flange is reduced to b_e, the centroid shifts away from it,
    and S_e is taken to the compression flange.

    Args:
        properties: derived SectionProperties.
        axis: axis of bending.
        yield_stress: F_y (ksi).
        youngs_modulus: E (ksi).
    """
    major = axis is Axis.MAJOR
    t = properties.design_wall_thickness
    area = properties.area
    inertia = properties.second_moment_x if major else properties.second_moment_y
    depth = properties.box_height if major else properties.box_width
    slenderness = properties.flange_slenderness if major else properties.web_slenderness
    if None in (t, area, inertia, depth, slenderness):
        return None

    b_e = calculate_effective_width(
        slenderness, t, yield_stress, youngs_modulus, properties.box_section
    )
    b = slenderness * t
    if b_e >= b:
        return inertia / (depth / 2)
    lost_area = (b - b_e) * t
    lever = depth / 2 - t / 2
    reduced_area = area - lost_area
    shift = -lost_area * lever / reduced_area
    inertia_effective = inertia - lost_area * lever**2 - reduced_area * shift**2
    return inertia_effective / (depth / 2 - shift)


def calculate_limiting_yield_length(
    plastic_moment: FloatLike,
    radius_of_gyration: FloatLike,
    torsional_constant: FloatLike,
    area: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Limiting unbraced length for yielding, L_p (Eq. F7-10)."""
    return (
        0.13
        * float(youngs_modulus)
        * float(radius_of_gyration)
        * sqrt(float(torsional_constant) * float(area))
        / float(plastic_moment)
    )


def calculate_limiting_inelastic_length(
    yield_stress: FloatLike,
    section_modulus: FloatLike,
    radius_of_gyration: FloatLike,
    torsional_constant: FloatLike,
    area: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Limiting unbraced length for inelastic LTB, L_r (Eq. F7-11)."""
    return (
        2
        * float(youngs_modulus)
        * float(radius_of_gyration)
        * sqrt(float(torsional_constant) * float(area))
        / (0.7 * float(yield_stress) * float(section_modulus))
    )


def check_f7(
    properties: SectionProperties,
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    flange: Optional[SlendernessCheck],
    web: Optional[SlendernessCheck],
    axis: Axis = Axis.MAJOR,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F7 check.

    For minor-axis bending the roles of the two walls swap: the Table B4.1b
    "web" becomes the compression flange and vice versa.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        unbraced_length: L_b (in).
        flange: Table B4.1b check for the wall perpendicular to the bending axis.
        web: Table B4.1b check for the wall parallel to the bending plane.
        axis: axis of bending.
        ltb_modification_factor: C_b (Eq. F1-1).
        youngs_modulus: E (ksi).
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    C_b = float(ltb_modification_factor)
    L_b = float(unbraced_length)
    major = axis is Axis.MAJOR

    result = FlexuralStrength(
        section_reference="F7",
        title="Square and rectangular HSS and box sections",
        axis=axis,
    )
    modulus_name = "section_modulus_x" if major else "section_modulus_y"
    plastic_name = "plastic_modulus_x" if major else "plastic_modulus_y"
    absent = missing(properties, [modulus_name, plastic_name])
    if absent:
        result.add(LimitState("Y", "Yielding", "F7-1", note=missing_note(absent)))
        return result

    S = getattr(properties, modulus_name)
    Z = getattr(properties, plastic_name)
    M_p = F_y * Z  # Eq. F7-1
    result.intermediate_values["M_p"] = M_p
    result.add(LimitState("Y", "Yielding (plastic moment)", "F7-1", M_p))

    compression_wall = flange if major else web
    bending_wall = web if major else flange

    # --- F7.2 flange local buckling ---------------------------------------
    if compression_wall is None or not compression_wall.known:
        result.add(
            LimitState("FLB", "Flange local buckling", "F7-2", note=missing_note(["b/t"]))
        )
    elif compression_wall.classification is ElementClass.COMPACT:
        result.add(
            LimitState(
                "FLB",
                "Flange local buckling",
                "F7-2",
                note="compact flange — the limit state does not apply",
            )
        )
    elif compression_wall.classification is ElementClass.NONCOMPACT:
        M_n = min(
            interpolate_noncompact(
                M_p,
                F_y * S,
                compression_wall.slenderness,
                compression_wall.limiting_compact,
                compression_wall.limiting_noncompact,
            ),
            M_p,
        )
        result.add(LimitState("FLB", "Flange local buckling (noncompact)", "F7-2", M_n))
    else:
        S_e = calculate_effective_section_modulus(properties, axis, F_y, E)
        if S_e is None:
            result.add(
                LimitState(
                    "FLB",
                    "Flange local buckling (slender)",
                    "F7-3",
                    note=missing_note(["area", "design_wall_thickness", "box_height", "box_width"]),
                )
            )
        else:
            result.intermediate_values["S_e"] = S_e
            result.add(LimitState("FLB", "Flange local buckling (slender)", "F7-3", F_y * S_e))

    # --- F7.3 web local buckling ------------------------------------------
    if bending_wall is None or not bending_wall.known:
        result.add(LimitState("WLB", "Web local buckling", "F7-6", note=missing_note(["h/t"])))
    elif bending_wall.classification is ElementClass.COMPACT:
        result.add(
            LimitState(
                "WLB",
                "Web local buckling",
                "F7-6",
                note="compact web — the limit state does not apply",
            )
        )
    elif bending_wall.classification is ElementClass.NONCOMPACT:
        M_n = min(
            interpolate_noncompact(
                M_p,
                F_y * S,
                bending_wall.slenderness,
                bending_wall.limiting_compact,
                bending_wall.limiting_noncompact,
            ),
            M_p,
        )
        result.add(LimitState("WLB", "Web local buckling (noncompact)", "F7-6", M_n))
    elif compression_wall is not None and compression_wall.classification is ElementClass.SLENDER:
        result.add(
            LimitState(
                "WLB",
                "Web local buckling",
                "F7-7",
                note="slender web with a slender flange is not addressed by the Specification",
            )
        )
        result.warnings.append(
            "Box section with slender webs and slender flanges — outside the scope of F7.3."
        )
    elif compression_wall is None or not compression_wall.known:
        result.add(
            LimitState(
                "WLB", "Web local buckling (slender)", "F7-7", note=missing_note(["b/t"])
            )
        )
    else:
        # Eq. F7-7: R_pg per Eq. F5-6 with a_w = 2ht/(bt)
        a_w = min(2 * bending_wall.slenderness / max(compression_wall.slenderness, 1e-9), 10.0)
        R_pg = calculate_bending_strength_reduction_factor(
            a_w, bending_wall.slenderness, F_y, E
        )
        result.intermediate_values.update({"R_pg": R_pg, "a_w": a_w})
        result.add(
            LimitState(
                "WLB",
                "Web local buckling (slender)",
                "F7-7",
                R_pg * F_y * S,
                note="R_pg per Eq. F5-6 with a_w = 2ht/(bt)",
            )
        )

    _add_lateral_torsional_buckling(result, properties, axis, F_y, S, M_p, L_b, C_b, E)
    return result


def _add_lateral_torsional_buckling(
    result: FlexuralStrength,
    properties: SectionProperties,
    axis: Axis,
    yield_stress: float,
    section_modulus: float,
    plastic_moment: float,
    unbraced_length: float,
    ltb_modification_factor: float,
    youngs_modulus: float,
) -> None:
    """Append the F7.4 lateral-torsional buckling limit state."""
    major = axis is Axis.MAJOR
    radius = properties.radius_of_gyration_y if major else properties.radius_of_gyration_x
    absent = missing(properties, ["torsional_constant", "area"])
    if radius is None:
        absent.append("radius_of_gyration_y" if major else "radius_of_gyration_x")
    if absent:
        result.add(
            LimitState(
                "LTB", "Lateral-torsional buckling", "F7-8", note=missing_note(absent)
            )
        )
        return

    J = properties.torsional_constant
    area = properties.area
    L_p = calculate_limiting_yield_length(plastic_moment, radius, J, area, youngs_modulus)
    L_r = calculate_limiting_inelastic_length(
        yield_stress, section_modulus, radius, J, area, youngs_modulus
    )
    result.intermediate_values.update({"L_p": L_p, "L_r": L_r})

    if unbraced_length <= L_p:
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling",
                "F7-8",
                note="L_b <= L_p — the limit state does not apply",
            )
        )
    elif unbraced_length <= L_r:
        M_n = ltb_modification_factor * (
            plastic_moment
            - (plastic_moment - 0.7 * yield_stress * section_modulus)
            * (unbraced_length - L_p)
            / (L_r - L_p)
        )  # F7-8
        result.add(
            LimitState(
                "LTB", "Lateral-torsional buckling (inelastic)", "F7-8", min(M_n, plastic_moment)
            )
        )
    else:
        M_n = (
            2
            * youngs_modulus
            * ltb_modification_factor
            * sqrt(J * area)
            / (unbraced_length / radius)
        )  # F7-9
        result.add(
            LimitState(
                "LTB", "Lateral-torsional buckling (elastic)", "F7-9", min(M_n, plastic_moment)
            )
        )
