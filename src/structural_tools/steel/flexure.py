"""Steel flexural design per AISC 360-22 Chapter F (Section F2).

Pure equation functions map 1:1 to the Specification and take plain floats.
calculate_nominal_flexural_strength accepts a WSection and derives intermediates.
Units: kip-inch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import TYPE_CHECKING

import numpy as np

from ..typing import FloatLike  # NOTE: two dots now
from .constants import YOUNGS_MODULUS_KSI

if TYPE_CHECKING:
    from .section import WSection


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
    I-shape, c = 1.0 (Eq. F2-8a) — handled on WSection.

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


def calculate_lateral_torsional_buckling_modification_factor(
    max_moment: FloatLike,
    quarter_moment: FloatLike,
    mid_moment: FloatLike,
    three_quarter_moment: FloatLike,
) -> float:
    """LTB modification factor C_b via the quarter-point method (Eq. F1-1).
    Absolute moment values are used internally.
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
    """Equivalent extreme-fiber stress for INELASTIC LTB, L_p < L_b <= L_r
    (Eq. F2-2 expressed as M_n / S_x). E cancels out of this equation.
    Capped at M_p / S_x so M_n never exceeds M_p.
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
    return C_b * np.pi**2 * E / slenderness**2 * sqrt(1 + 0.078 * (J * c) / (S_x * h_o) * slenderness**2)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class LateralTorsionalBucklingResult:
    """Result of an AISC 360-22 F2 lateral-torsional buckling check.

    Moments in kip-in, lengths in inches (E in ksi).
    """

    nominal_moment: float  # M_n
    plastic_moment: float  # M_p
    limiting_yield_length: float  # L_p
    limiting_inelastic_length: float  # L_r
    unbraced_length: float  # L_b
    effective_radius_of_gyration: float  # r_ts (derived)
    torsional_coefficient: float  # c    (derived)
    region: str  # "plastic" | "inelastic" | "elastic"


def calculate_nominal_flexural_strength(
    section: WSection,
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> LateralTorsionalBucklingResult:
    """Nominal flexural strength M_n for LTB of a compact doubly symmetric
    I-shape or channel bent about its major axis (AISC 360-22 Section F2).

    r_ts, c, M_p, L_p, and L_r are derived from `section` and the material.
    Assumes a COMPACT section (no F3/F4/F5 local buckling). Apply phi=0.90
    (LRFD) or Omega=1.67 (ASD) to M_n at the call site.
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    C_b = float(ltb_modification_factor)
    L_b = float(unbraced_length)

    Z_x = section.plastic_section_modulus
    S_x = section.section_modulus_x_axis
    r_y = section.radius_of_gyration_y_axis
    J = section.torsional_constant
    h_o = section.flange_centroid_distance
    r_ts = section.effective_radius_of_gyration  # derived once, on the section
    c = section.torsional_coefficient  # derived once, on the section

    M_p = calculate_plastic_moment(F_y, Z_x)  # F2-1
    L_p = calculate_limiting_yield_length(F_y, r_y, E)  # F2-5
    L_r = calculate_limiting_inelastic_length(F_y, r_ts, J, c, S_x, h_o, E)  # F2-6

    if L_b <= L_p:
        region, M_n = "plastic", M_p  # F2-1
    elif L_b <= L_r:
        region = "inelastic"  # F2-2
        F_n = calculate_inelastic_lateral_torsional_buckling_stress(F_y, Z_x, S_x, L_b, L_p, L_r, C_b)
        M_n = min(F_n * S_x, M_p)
    else:
        region = "elastic"  # F2-3 / F2-4
        F_cr = calculate_elastic_lateral_torsional_buckling_stress(L_b, r_ts, J, c, S_x, h_o, C_b, E)
        M_n = min(F_cr * S_x, M_p)

    return LateralTorsionalBucklingResult(
        nominal_moment=M_n,
        plastic_moment=M_p,
        limiting_yield_length=L_p,
        limiting_inelastic_length=L_r,
        unbraced_length=L_b,
        effective_radius_of_gyration=r_ts,
        torsional_coefficient=c,
        region=region,
    )
