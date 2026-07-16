"""Steel gravity calculations using AISC 15th Edition"""

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
    youngs_modulus: FloatLike,
    effective_length: FloatLike,
    radius_of_gyration: FloatLike,
) -> float:
    E = float(youngs_modulus)
    L_c = float(effective_length)
    r = float(radius_of_gyration)

    return np.pi**2 * E / (L_c / r) ** 2


def calculate_critical_stress(
    yield_stress: FloatLike,
    youngs_modulus: FloatLike,
    effective_length: FloatLike,
    radius_of_gyration: FloatLike,
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
