"""Steel gravity calculations using AISC 15th Edition"""

from math import sqrt

import numpy as np

from .typing import FloatLike


def calculate_slenderness_ratio(
    effective_length: FloatLike,
    radius_of_gyration: FloatLike,
) -> float:
    L_c = float(effective_length)
    r = float(radius_of_gyration)
    return L_c / r


def calculate_euler_buckling_stress(
    effective_length: FloatLike,
    radius_of_gyration: FloatLike,
    youngs_modulus: FloatLike = 29000,
) -> float:
    E = float(youngs_modulus)
    L_c = float(effective_length)
    r = float(radius_of_gyration)

    return np.pi**2 * E / (L_c / r) ** 2


def calculate_critical_stress(
    yield_stress: FloatLike,
    effective_length: FloatLike,
    radius_of_gyration: FloatLike,
    youngs_modulus: FloatLike = 29000,
) -> float:
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    L_c = float(effective_length)
    r = float(radius_of_gyration)
    F_e = calculate_euler_buckling_stress(youngs_modulus=E, effective_length=L_c, radius_of_gyration=r)
    if F_y / F_e <= 2.25:
        return (0.658 ** (F_y / F_e)) * F_y
    else:
        return 0.877 * F_e


def calculate_limiting_yield_length(
    yield_stress: FloatLike,
    radius_of_gyration_y_axis: FloatLike,
    youngs_modulus: FloatLike = 29000,
) -> float:
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    r_y = float(radius_of_gyration_y_axis)

    return 1.76 * r_y * sqrt(E / F_y)


def calculate_effective_radius_of_gyration(
    second_moment_of_area_y_axis: FloatLike,
    warping_constant: FloatLike,
    section_modulus_x_axis: FloatLike,
):
    """Calculate the effective radius of gyration, r_ts.

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
    second_moment_fo_area_y_axis: FloatLike,
    warping_constant: FloatLike,
) -> float:
    h_o = float(flange_centroid_distance)
    I_y = float(second_moment_fo_area_y_axis)
    C_w = float(warping_constant)

    return h_o / 2 * sqrt(I_y / C_w)


def calculate_inelastic_lateral_torsional_buckling_stress(
    unbraced_length: FloatLike,
    youngs_modulus: FloatLike = 29000,
) -> float:
    pass


def calculate_elastic_lateral_torsional_buckling_stress(
    unbraced_length: FloatLike,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = 29000,
) -> float:
    pass


"""Steel gravity calculations using AISC 15th/16th Edition (360-16/360-22)."""

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .typing import FloatLike


# --- (your existing functions above: slenderness, Euler, Fcr, Lp, r_ts, c) ---


def calculate_plastic_moment(
    yield_stress: FloatLike,
    plastic_section_modulus: FloatLike,
) -> float:
    """Plastic moment M_p = F_y * Z_x  (AISC 360-22 Eq. F2-1).

    Returns M_p in kip-in when F_y is ksi and Z_x is in^3.
    """
    return float(yield_stress) * float(plastic_section_modulus)


def calculate_limiting_inelastic_length(
    yield_stress: FloatLike,
    effective_radius_of_gyration: FloatLike,
    torsional_constant: FloatLike,
    torsional_coefficient: FloatLike,
    section_modulus_x_axis: FloatLike,
    flange_centroid_distance: FloatLike,
    youngs_modulus: FloatLike = 29000,
) -> float:
    """Limiting laterally unbraced length for the limit state of inelastic LTB,
    L_r  (AISC 360-22 Eq. F2-6).

    Args:
        yield_stress: F_y
        effective_radius_of_gyration: r_ts  (see calculate_effective_radius_of_gyration)
        torsional_constant: J
        torsional_coefficient: c  (1.0 for doubly symmetric I; F2-8b for channels)
        section_modulus_x_axis: S_x
        flange_centroid_distance: h_o
        youngs_modulus: E

    Returns: L_r  (inches)
    """
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
    """Lateral-torsional buckling modification factor C_b via the quarter-point
    method (AISC 360-22 Eq. F1-1). Absolute moment values are used internally.

    Args:
        max_moment:          |M_max|  over the unbraced segment
        quarter_moment:      |M_A|    at the 1/4 point
        mid_moment:          |M_B|    at the mid point
        three_quarter_moment:|M_C|    at the 3/4 point

    Returns: C_b  (dimensionless)
    """
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
    """Equivalent extreme-fiber stress for INELASTIC LTB, valid L_p < L_b <= L_r.

    This is AISC 360-22 Eq. F2-2 expressed as a stress (M_n / S_x) so it composes
    with the elastic branch via M_n = F * S_x. The result is capped at the plastic
    stress M_p / S_x so that M_n never exceeds M_p.

    Args:
        yield_stress: F_y
        plastic_section_modulus: Z_x
        section_modulus_x_axis: S_x
        unbraced_length: L_b
        limiting_yield_length: L_p  (Eq. F2-5)
        limiting_inelastic_length: L_r  (Eq. F2-6)
        ltb_modification_factor: C_b

    Returns: equivalent stress M_n / S_x  (ksi)
    """
    F_y = float(yield_stress)
    Z_x = float(plastic_section_modulus)
    S_x = float(section_modulus_x_axis)
    L_b = float(unbraced_length)
    L_p = float(limiting_yield_length)
    L_r = float(limiting_inelastic_length)
    C_b = float(ltb_modification_factor)

    F_p = F_y * Z_x / S_x  # M_p / S_x, the plastic-stress equivalent
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
    youngs_modulus: FloatLike = 29000,
) -> float:
    """Critical stress for ELASTIC LTB, F_cr, valid L_b > L_r
    (AISC 360-22 Eq. F2-4).

    Args:
        unbraced_length: L_b
        effective_radius_of_gyration: r_ts
        torsional_constant: J
        torsional_coefficient: c
        section_modulus_x_axis: S_x
        flange_centroid_distance: h_o
        ltb_modification_factor: C_b
        youngs_modulus: E

    Returns: F_cr  (ksi)
    """
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
