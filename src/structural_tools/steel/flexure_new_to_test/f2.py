"""AISC 360-22 Section F2 — doubly symmetric compact I-shapes and channels,
major axis. Limit states: yielding, lateral-torsional buckling.

Units: kip-inch.
"""

from __future__ import annotations

from math import pi, sqrt

from ...typing import FloatLike
from .common import Axis, FlexuralStrength, LimitState, missing, missing_note
from ..constants import YOUNGS_MODULUS_KSI
from .properties import SectionProperties


def calculate_effective_radius_of_gyration(
    second_moment_of_area_y_axis: FloatLike,
    warping_constant: FloatLike,
    section_modulus_x_axis: FloatLike,
) -> float:
    """Effective radius of gyration, r_ts (Eq. F2-7).

    Args:
        second_moment_of_area_y_axis: I_y
        warping_constant: C_w
        section_modulus_x_axis: S_x
    """
    I_y = float(second_moment_of_area_y_axis)
    C_w = float(warping_constant)
    S_x = float(section_modulus_x_axis)
    return sqrt(sqrt(I_y * C_w) / S_x)


def calculate_torsional_coefficient(
    flange_centroid_distance: FloatLike,
    second_moment_of_area_y_axis: FloatLike,
    warping_constant: FloatLike,
) -> float:
    """Coefficient c for a channel (Eq. F2-8b). For a doubly symmetric
    I-shape, c = 1.0 (Eq. F2-8a) — handled on the section.

    Args:
        flange_centroid_distance: h_o
        second_moment_of_area_y_axis: I_y
        warping_constant: C_w
    """
    h_o = float(flange_centroid_distance)
    I_y = float(second_moment_of_area_y_axis)
    C_w = float(warping_constant)
    return h_o / 2 * sqrt(I_y / C_w)


def calculate_plastic_moment(
    yield_stress: FloatLike,
    plastic_section_modulus: FloatLike,
) -> float:
    """Plastic moment M_p = F_y * Z_x (Eq. F2-1)."""
    return float(yield_stress) * float(plastic_section_modulus)


def calculate_limiting_yield_length(
    yield_stress: FloatLike,
    radius_of_gyration_y_axis: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Limiting unbraced length for the limit state of yielding, L_p (Eq. F2-5)."""
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    r_y = float(radius_of_gyration_y_axis)
    return 1.76 * r_y * sqrt(E / F_y)


def calculate_limiting_inelastic_length(
    yield_stress: FloatLike,
    effective_radius_of_gyration: FloatLike,
    torsional_constant: FloatLike,
    torsional_coefficient: FloatLike,
    section_modulus_x_axis: FloatLike,
    flange_centroid_distance: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Limiting unbraced length for inelastic LTB, L_r (Eq. F2-6)."""
    F_y = float(yield_stress)
    r_ts = float(effective_radius_of_gyration)
    J = float(torsional_constant)
    c = float(torsional_coefficient)
    S_x = float(section_modulus_x_axis)
    h_o = float(flange_centroid_distance)
    E = float(youngs_modulus)

    F_L = 0.7 * F_y
    beta = (J * c) / (S_x * h_o)
    return 1.95 * r_ts * (E / F_L) * sqrt(beta + sqrt(beta**2 + 6.76 * (F_L / E) ** 2))


def calculate_inelastic_lateral_torsional_buckling_moment(
    plastic_moment: FloatLike,
    yield_stress: FloatLike,
    section_modulus_x_axis: FloatLike,
    unbraced_length: FloatLike,
    limiting_yield_length: FloatLike,
    limiting_inelastic_length: FloatLike,
    ltb_modification_factor: FloatLike = 1.0,
) -> float:
    """Nominal moment for INELASTIC LTB, L_p < L_b <= L_r (Eq. F2-2), capped at M_p."""
    M_p = float(plastic_moment)
    F_y = float(yield_stress)
    S_x = float(section_modulus_x_axis)
    L_b = float(unbraced_length)
    L_p = float(limiting_yield_length)
    L_r = float(limiting_inelastic_length)
    C_b = float(ltb_modification_factor)

    M_n = C_b * (M_p - (M_p - 0.7 * F_y * S_x) * (L_b - L_p) / (L_r - L_p))
    return min(M_n, M_p)


def calculate_elastic_lateral_torsional_buckling_stress(
    unbraced_length: FloatLike,
    effective_radius_of_gyration: FloatLike,
    torsional_constant: FloatLike,
    torsional_coefficient: FloatLike,
    section_modulus_x_axis: FloatLike,
    flange_centroid_distance: FloatLike,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Critical stress for ELASTIC LTB, F_cr, valid L_b > L_r (Eq. F2-4)."""
    L_b = float(unbraced_length)
    r_ts = float(effective_radius_of_gyration)
    J = float(torsional_constant)
    c = float(torsional_coefficient)
    S_x = float(section_modulus_x_axis)
    h_o = float(flange_centroid_distance)
    C_b = float(ltb_modification_factor)
    E = float(youngs_modulus)

    slenderness = L_b / r_ts
    return (
        C_b
        * pi**2
        * E
        / slenderness**2
        * sqrt(1 + 0.078 * (J * c) / (S_x * h_o) * slenderness**2)
    )


def check_f2(
    properties: SectionProperties,
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F2 check: yielding and lateral-torsional buckling.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        unbraced_length: L_b (in).
        ltb_modification_factor: C_b (Eq. F1-1).
        youngs_modulus: E (ksi).
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    C_b = float(ltb_modification_factor)
    L_b = float(unbraced_length)

    result = FlexuralStrength(
        section_reference="F2",
        title="Doubly symmetric compact I-shaped members and channels bent about their major axis",
        axis=Axis.MAJOR,
    )
    absent = missing(
        properties,
        [
            "plastic_modulus_x",
            "section_modulus_x",
            "radius_of_gyration_y",
            "effective_radius_of_gyration",
            "torsional_constant",
            "flange_centroid_distance",
        ],
    )
    if absent:
        result.add(LimitState("Y", "Yielding", "F2-1", note=missing_note(absent)))
        return result

    Z_x = properties.plastic_modulus_x
    S_x = properties.section_modulus_x
    r_y = properties.radius_of_gyration_y
    r_ts = properties.effective_radius_of_gyration
    J = properties.torsional_constant
    c = properties.torsional_coefficient or 1.0
    h_o = properties.flange_centroid_distance

    M_p = calculate_plastic_moment(F_y, Z_x)  # F2-1
    L_p = calculate_limiting_yield_length(F_y, r_y, E)  # F2-5
    L_r = calculate_limiting_inelastic_length(F_y, r_ts, J, c, S_x, h_o, E)  # F2-6
    result.intermediate_values.update(
        {"M_p": M_p, "L_p": L_p, "L_r": L_r, "r_ts": r_ts, "c": c}
    )

    result.add(LimitState("Y", "Yielding (plastic moment)", "F2-1", M_p))

    if L_b <= L_p:
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling",
                "F2-2",
                note="L_b <= L_p — the limit state does not apply",
            )
        )
    elif L_b <= L_r:
        M_n = calculate_inelastic_lateral_torsional_buckling_moment(
            M_p, F_y, S_x, L_b, L_p, L_r, C_b
        )
        result.add(LimitState("LTB", "Lateral-torsional buckling (inelastic)", "F2-2", M_n))
    else:
        F_cr = calculate_elastic_lateral_torsional_buckling_stress(
            L_b, r_ts, J, c, S_x, h_o, C_b, E
        )
        result.intermediate_values["F_cr"] = F_cr
        result.add(
            LimitState(
                "LTB", "Lateral-torsional buckling (elastic)", "F2-3 / F2-4", min(F_cr * S_x, M_p)
            )
        )
    return result
