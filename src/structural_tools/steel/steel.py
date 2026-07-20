"""Steel gravity calculations using AISC 15th/16th Edition (360-16/360-22).

All defaults assume kip-inch units (USA practice):
    F_y [ksi], lengths [in], I/J [in^4], S/Z [in^3], moments [kip-in].
"""

from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np

from .typing import FloatLike

#: Modulus of elasticity of structural steel, E (ksi). AISC 360-22 uses 29,000 ksi.
YOUNGS_MODULUS_KSI: float = 29000


def calculate_slenderness_ratio(
    effective_length: FloatLike,
    radius_of_gyration: FloatLike,
) -> float:
    return float(effective_length) / float(radius_of_gyration)


def calculate_euler_buckling_stress(
    effective_length: FloatLike,
    radius_of_gyration: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    E = float(youngs_modulus)
    L_c = float(effective_length)
    r = float(radius_of_gyration)
    return np.pi**2 * E / (L_c / r) ** 2


def calculate_critical_stress(
    yield_stress: FloatLike,
    effective_length: FloatLike,
    radius_of_gyration: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    F_e = calculate_euler_buckling_stress(
        effective_length=effective_length,
        radius_of_gyration=radius_of_gyration,
        youngs_modulus=E,
    )
    if F_y / F_e <= 2.25:
        return (0.658 ** (F_y / F_e)) * F_y
    return 0.877 * F_e


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

    Returns: r_ts
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
    I-shape, c = 1.0 (Eq. F2-8a) — handled in the orchestrator.

    Args:
        flange_centroid_distance: h_o
        second_moment_of_area_y_axis: I_y
        warping_constant: C_w

    Returns: c
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


def calculate_lateral_torsional_buckling_modification_factor(
    max_moment: FloatLike,
    quarter_moment: FloatLike,
    mid_moment: FloatLike,
    three_quarter_moment: FloatLike,
) -> float:
    """LTB modification factor C_b via the quarter-point method (Eq. F1-1)."""
    M_max = abs(float(max_moment))
    M_a = abs(float(quarter_moment))
    M_b = abs(float(mid_moment))
    M_c = abs(float(three_quarter_moment))
    return 12.5 * M_max / (2.5 * M_max + 3 * M_a + 4 * M_b + 3 * M_c)


def calculate_inelastic_lateral_torsional_buckling_stress(
    yield_stress: FloatLike,
    plastic_section_modulus: FloatLike,
    section_modulus_x_axis: FloatLike,
    unbraced_length: FloatLike,
    limiting_yield_length: FloatLike,
    limiting_inelastic_length: FloatLike,
    ltb_modification_factor: FloatLike = 1.0,
) -> float:
    """Equivalent extreme-fiber stress for INELASTIC LTB, L_p < L_b <= L_r
    (Eq. F2-2 as M_n / S_x). E cancels out of this equation. Capped at M_p / S_x.
    """
    F_y = float(yield_stress)
    Z_x = float(plastic_section_modulus)
    S_x = float(section_modulus_x_axis)
    L_b = float(unbraced_length)
    L_p = float(limiting_yield_length)
    L_r = float(limiting_inelastic_length)
    C_b = float(ltb_modification_factor)

    F_p = F_y * Z_x / S_x
    F_n = C_b * (F_p - (F_p - 0.7 * F_y) * (L_b - L_p) / (L_r - L_p))
    return min(F_n, F_p)


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
    """Critical stress for ELASTIC LTB, F_cr, L_b > L_r (Eq. F2-4)."""
    L_b = float(unbraced_length)
    r_ts = float(effective_radius_of_gyration)
    J = float(torsional_constant)
    c = float(torsional_coefficient)
    S_x = float(section_modulus_x_axis)
    h_o = float(flange_centroid_distance)
    C_b = float(ltb_modification_factor)
    E = float(youngs_modulus)

    slenderness = L_b / r_ts
    return C_b * np.pi**2 * E / slenderness**2 * sqrt(1 + 0.078 * (J * c) / (S_x * h_o) * slenderness**2)
