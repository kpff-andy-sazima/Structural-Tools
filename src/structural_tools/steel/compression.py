"""Steel compression member calculations (AISC 360-22 Chapter E).

Units: kip-inch. E defaults to YOUNGS_MODULUS_KSI.
"""

import numpy as np
from ..typing import FloatLike  # NOTE: two dots now
from .constants import YOUNGS_MODULUS_KSI


# calculate_slenderness_ratio, calculate_euler_buckling_stress,
# calculate_critical_stress  — bodies unchanged from the previous steel.py
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
