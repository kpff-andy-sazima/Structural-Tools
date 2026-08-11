"""AISC 360-22 Section F9 — tees and double angles loaded in the plane of
symmetry. Limit states: yielding, lateral-torsional buckling, flange local
buckling, and local buckling of tee stems / double-angle web legs.

C_b does not appear anywhere in F9 of the 2022 Specification: Eq. F9-10 for
M_cr is written without it, so no moment-gradient factor is applied here.

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
    ShapeGroup,
    StemOrientation,
    interpolate_noncompact,
    missing,
    missing_note,
)
from ..constants import YOUNGS_MODULUS_KSI
from .properties import SectionProperties


def calculate_yield_moment(yield_stress: FloatLike, section_modulus_x_axis: FloatLike) -> float:
    """Yield moment about the axis of bending, M_y = F_y S_x (Eq. F9-3)."""
    return float(yield_stress) * float(section_modulus_x_axis)


def calculate_plastic_moment(
    yield_stress: FloatLike,
    plastic_section_modulus_x_axis: FloatLike,
    yield_moment: FloatLike,
    stem: StemOrientation,
    double_angle: bool = False,
) -> float:
    """M_p for a tee or double angle (Eq. F9-2, F9-4, F9-5).

    Args:
        yield_stress: F_y
        plastic_section_modulus_x_axis: Z_x
        yield_moment: M_y per Eq. F9-3
        stem: state of the stem / web leg over the unbraced length
        double_angle: True for 2L sections
    """
    M_y = float(yield_moment)
    if stem is StemOrientation.TENSION:
        return min(float(yield_stress) * float(plastic_section_modulus_x_axis), 1.6 * M_y)  # F9-2
    return 1.5 * M_y if double_angle else M_y  # F9-5 / F9-4


def calculate_limiting_yield_length(
    yield_stress: FloatLike,
    radius_of_gyration_y_axis: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Limiting unbraced length for yielding, L_p (Eq. F9-8)."""
    return 1.76 * float(radius_of_gyration_y_axis) * sqrt(
        float(youngs_modulus) / float(yield_stress)
    )


def calculate_limiting_inelastic_length(
    yield_stress: FloatLike,
    second_moment_of_area_y_axis: FloatLike,
    torsional_constant: FloatLike,
    section_modulus_x_axis: FloatLike,
    depth: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Limiting unbraced length for inelastic LTB, L_r (Eq. F9-9)."""
    F_y = float(yield_stress)
    I_y = float(second_moment_of_area_y_axis)
    J = float(torsional_constant)
    S_x = float(section_modulus_x_axis)
    d = float(depth)
    E = float(youngs_modulus)
    return 1.95 * (E / F_y) * (sqrt(I_y * J) / S_x) * sqrt(2.36 * (F_y / E) * (d * S_x / J) + 1)


def calculate_elastic_buckling_moment(
    unbraced_length: FloatLike,
    second_moment_of_area_y_axis: FloatLike,
    torsional_constant: FloatLike,
    depth: FloatLike,
    stem: StemOrientation,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Elastic LTB moment M_cr (Eq. F9-10) with B per Eq. F9-11 / F9-12.

    B is negative when the stem or web leg is in compression anywhere along
    the unbraced length.

    Args:
        unbraced_length: L_b
        second_moment_of_area_y_axis: I_y
        torsional_constant: J
        depth: d, depth of tee or width of the web leg
        stem: state of the stem / web leg
        youngs_modulus: E (ksi)
    """
    L_b = float(unbraced_length)
    I_y = float(second_moment_of_area_y_axis)
    J = float(torsional_constant)
    d = float(depth)
    E = float(youngs_modulus)
    sign = -1.0 if stem is StemOrientation.COMPRESSION else 1.0
    B = sign * 2.3 * (d / L_b) * sqrt(I_y / J)
    return 1.95 * (E / L_b) * sqrt(I_y * J) * (B + sqrt(1 + B**2))


def calculate_stem_local_buckling_stress(
    stem_slenderness: FloatLike,
    yield_stress: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Critical stress for a tee stem in flexural compression (Eq. F9-17 … F9-19)."""
    lam = float(stem_slenderness)
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    root = sqrt(E / F_y)
    if lam <= 0.84 * root:
        return F_y  # F9-17
    if lam <= 1.52 * root:
        return (1.43 - 0.515 * lam * sqrt(F_y / E)) * F_y  # F9-18
    return 1.52 * E / lam**2  # F9-19


def check_f9(
    properties: SectionProperties,
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    flange: Optional[SlendernessCheck],
    stem_check: Optional[SlendernessCheck],
    stem: StemOrientation = StemOrientation.TENSION,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F9 check.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        unbraced_length: L_b (in).
        flange: Table B4.1b check for the flange / flange leg.
        stem_check: Table B4.1b check for the stem (d/t_w).
        stem: whether the stem or web leg is in tension or compression.
        youngs_modulus: E (ksi).
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    L_b = float(unbraced_length)
    double_angle = properties.group is ShapeGroup.DOUBLE_ANGLE

    result = FlexuralStrength(
        section_reference="F9",
        title="Tees and double angles loaded in the plane of symmetry",
        axis=Axis.MAJOR,
    )
    absent = missing(
        properties,
        ["plastic_modulus_x", "section_modulus_x", "second_moment_y", "torsional_constant", "depth"],
    )
    if absent:
        result.add(LimitState("Y", "Yielding", "F9-1", note=missing_note(absent)))
        return result

    S_x = properties.section_modulus_x
    S_xc = properties.section_modulus_compression or S_x
    I_y = properties.second_moment_y
    J = properties.torsional_constant
    d = properties.depth

    M_y = calculate_yield_moment(F_y, S_x)  # F9-3
    M_p = calculate_plastic_moment(F_y, properties.plastic_modulus_x, M_y, stem, double_angle)
    result.intermediate_values.update({"M_p": M_p, "M_y": M_y})
    label = "stem/web leg in compression" if stem is StemOrientation.COMPRESSION else (
        "stem/web leg in tension"
    )
    result.add(LimitState("Y", f"Yielding ({label})", "F9-1", M_p))

    # --- F9.2 lateral-torsional buckling ----------------------------------
    if L_b <= 0:
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling",
                "F9-6",
                note="continuously braced — the limit state does not apply",
            )
        )
    elif stem is StemOrientation.COMPRESSION:
        M_cr = calculate_elastic_buckling_moment(L_b, I_y, J, d, stem, E)
        result.intermediate_values["M_cr"] = M_cr
        if double_angle:
            M_n = _double_angle_web_leg_moment(M_cr, M_y)
            result.add(
                LimitState(
                    "LTB",
                    "Lateral-torsional buckling (web leg in compression)",
                    "F9-10 with F10-2 / F10-3",
                    M_n,
                )
            )
        else:
            result.add(
                LimitState(
                    "LTB",
                    "Lateral-torsional buckling (stem in compression)",
                    "F9-13",
                    min(M_cr, M_y),
                )
            )
    else:
        absent = missing(properties, ["radius_of_gyration_y"])
        if absent:
            result.add(
                LimitState(
                    "LTB", "Lateral-torsional buckling", "F9-8", note=missing_note(absent)
                )
            )
        else:
            L_p = calculate_limiting_yield_length(F_y, properties.radius_of_gyration_y, E)
            L_r = calculate_limiting_inelastic_length(F_y, I_y, J, S_x, d, E)
            result.intermediate_values.update({"L_p": L_p, "L_r": L_r})
            if L_b <= L_p:
                result.add(
                    LimitState(
                        "LTB",
                        "Lateral-torsional buckling",
                        "F9-6",
                        note="L_b <= L_p — the limit state does not apply",
                    )
                )
            elif L_b <= L_r:
                M_n = M_p - (M_p - M_y) * (L_b - L_p) / (L_r - L_p)  # F9-6
                result.add(
                    LimitState("LTB", "Lateral-torsional buckling (inelastic)", "F9-6", M_n)
                )
            else:
                M_cr = calculate_elastic_buckling_moment(L_b, I_y, J, d, stem, E)
                result.intermediate_values["M_cr"] = M_cr
                result.add(
                    LimitState("LTB", "Lateral-torsional buckling (elastic)", "F9-7 / F9-10", M_cr)
                )

    # --- F9.3 flange local buckling ---------------------------------------
    if flange is None or not flange.known:
        result.add(
            LimitState(
                "FLB",
                "Flange local buckling",
                "F9-14",
                note=missing_note(["flange_slenderness"]),
            )
        )
    elif double_angle:
        result.add(
            LimitState(
                "FLB",
                "Flange leg local buckling",
                "F10.3",
                note="double angles — check the flange legs per Section F10.3",
            )
        )
    elif flange.classification is ElementClass.COMPACT:
        result.add(
            LimitState(
                "FLB",
                "Flange local buckling",
                "F9-14",
                note="compact flange — the limit state does not apply",
            )
        )
    elif flange.classification is ElementClass.NONCOMPACT:
        M_n = min(
            interpolate_noncompact(
                M_p,
                0.7 * F_y * S_xc,
                flange.slenderness,
                flange.limiting_compact,
                flange.limiting_noncompact,
            ),
            1.6 * M_y,
        )  # F9-14
        result.add(LimitState("FLB", "Flange local buckling (noncompact)", "F9-14", M_n))
    else:
        M_n = 0.7 * E * S_xc / flange.slenderness**2  # F9-15
        result.add(LimitState("FLB", "Flange local buckling (slender)", "F9-15", M_n))

    # --- F9.4 stem / web leg local buckling -------------------------------
    if stem is not StemOrientation.COMPRESSION:
        result.add(
            LimitState(
                "SLB",
                "Stem local buckling",
                "F9-16",
                note="stem in tension — the limit state does not apply",
            )
        )
    elif double_angle:
        result.add(
            LimitState(
                "SLB",
                "Web leg local buckling",
                "F10.3",
                note="double angles — check the web legs per Section F10.3",
            )
        )
    elif stem_check is None or stem_check.slenderness is None:
        result.add(
            LimitState("SLB", "Stem local buckling", "F9-16", note=missing_note(["d/t_w"]))
        )
    else:
        F_cr = calculate_stem_local_buckling_stress(stem_check.slenderness, F_y, E)
        result.intermediate_values["F_cr_stem"] = F_cr
        result.add(
            LimitState(
                "SLB",
                "Stem local buckling in flexural compression",
                "F9-16 … F9-19",
                F_cr * S_x,
            )
        )
    return result


def _double_angle_web_leg_moment(elastic_moment: float, yield_moment: float) -> float:
    """M_n for a double-angle web leg in compression (F9.2(b)(2)).

    Uses Eq. F10-2 / F10-3 with M_cr from Eq. F9-10 and M_y from Eq. F9-3.
    """
    from .f10 import calculate_lateral_torsional_buckling_moment

    return calculate_lateral_torsional_buckling_moment(elastic_moment, yield_moment)
