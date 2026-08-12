"""Steel flexural design per AISC 360-22 Chapter F (Section F2).

Pure equation functions map 1:1 to the Specification and take plain floats.
calculate_nominal_flexural_strength accepts a WSection and derives intermediates.
Units: kip-inch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

from ..typing import FloatLike
from .constants import YOUNGS_MODULUS_KSI

if TYPE_CHECKING:
    from .section import WSection


class Bracing(Enum):
    """Lateral bracing condition for a single flange.

    CONTINUOUS: braced along the entire length (e.g. by a floor slab/deck),
        so the flange cannot buckle laterally and develops M_p (L_b = 0).
    UNBRACED: no bracing. L_b = full span
    TENSION_ONLY: braced flange is the tension flange
    """

    CONTINUOUS = "continuous"
    UNBRACED = "unbraced"
    TENSION_ONLY = "tension_only"


# A flange's bracing is either explicit brace-point x-coordinates or a
# Bracing member (currently just CONTINUOUS).
BracePoints = Union[Sequence[FloatLike], Bracing]


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
    plastic_section_modulus_x_axis: FloatLike,
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
    Z_x = float(plastic_section_modulus_x_axis)
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

    Z_x = section.plastic_section_modulus_x_axis
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
        region, M_n = "Plastic", M_p  # F2-1
    elif L_b <= L_r:
        region = "Inelastic LTB"  # F2-2
        F_n = calculate_inelastic_lateral_torsional_buckling_stress(F_y, Z_x, S_x, L_b, L_p, L_r, C_b)
        M_n = min(F_n * S_x, M_p)
    else:
        region = "Elastic LTB"  # F2-3 / F2-4
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


def calculate_cb_from_moment_diagram(
    positions: Sequence[FloatLike] | NDArray[np.float64],
    moments: Sequence[FloatLike] | NDArray[np.float64],
) -> float:
    """C_b via the quarter-point method (Eq. F1-1) from a sampled diagram.

    Moments are interpolated at the quarter points of the sampled span and
    M_max is the largest absolute moment over the samples, so `positions`
    and `moments` may be irregularly spaced (e.g. straight from a solver).

    Args:
        positions: x-coordinates along the segment (consistent length unit).
        moments: bending moment at each x (consistent moment unit).
    """
    x = np.asarray(positions, dtype=float)
    M = np.asarray(moments, dtype=float)
    order = np.argsort(x)
    x, M = x[order], M[order]
    L = x[-1] - x[0]
    if L <= 0:
        return 1.0
    quarter, mid, three_quarter = np.interp(x[0] + L * np.array([0.25, 0.50, 0.75]), x, M)
    return calculate_lateral_torsional_buckling_modification_factor(np.max(np.abs(M)), quarter, mid, three_quarter)


def _slice_moment_segment(
    x: np.ndarray,
    M: np.ndarray,
    start: float,
    end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Samples of (x, M) on [start, end] with interpolated endpoints added."""
    mask = (x >= start) & (x <= end)
    xs, Ms = x[mask], M[mask]
    if xs.size == 0 or xs[0] > start:
        xs = np.insert(xs, 0, start)
        Ms = np.insert(Ms, 0, np.interp(start, x, M))
    if xs[-1] < end:
        xs = np.append(xs, end)
        Ms = np.append(Ms, np.interp(end, x, M))
    return xs, Ms


@dataclass
class FlexuralSegmentResult:
    """One unbraced-segment flexure check on a single compression flange.

    Moments in kip-in, lengths in inches.
    """

    flange: str  # "top" | "bottom"
    x_start: float
    x_end: float
    moment_demand: float  # M_u causing compression in this flange
    lateral_torsional_buckling: LateralTorsionalBucklingResult
    ltb_modification_factor: float  # C_b
    continuously_braced: bool = False
    governing: bool = False

    @property
    def demand_capacity_ratio(self) -> float:
        """M_u / M_n (apply phi=0.90 or Omega=1.67 at the call site)."""
        return self.moment_demand / self.lateral_torsional_buckling.nominal_moment


def _brace_segments(braces, x_min, x_max):
    """(start, end, continuous) unbraced segments for a flange.

    If `braces` is Bracing.CONTINUOUS, yields one whole-member segment flagged
    continuous. If `braces` is Bracing.UNBRACED, yields one whole-member segment
    flagged unbraced. Otherwise the member ends are added to the brace list and
    consecutive pairs are returned.
    """
    if braces is Bracing.CONTINUOUS:  # <- `is`, not string compare
        yield (x_min, x_max, True)
        return
    if braces is Bracing.UNBRACED:
        yield (x_min, x_max, False)
        return
    brace_x = sorted({x_min, x_max, *(float(b) for b in braces)})
    for start, end in zip(brace_x[:-1], brace_x[1:]):
        yield (start, end, False)


def evaluate_beam_flexure(
    section: WSection,
    yield_stress: FloatLike,
    positions: Sequence[FloatLike],
    moments: Sequence[FloatLike],
    top_flange_brace_points: Sequence[FloatLike] | Bracing,
    bottom_flange_brace_points: Sequence[FloatLike] | Bracing,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
    positive_moment_compresses: str = "bottom",
    zero_atol: float = 1e-6,
) -> list:
    """Full major-axis LTB check of a beam directly from a moment diagram.

    The compression flange is chosen automatically from the sign of the
    moment. Each flange is broken into unbraced segments by its own brace
    points (member ends added automatically); segments where that flange sees
    compression are checked with L_b = segment length, C_b from the
    quarter-point method over the segment, and the limit state (plastic /
    inelastic / elastic LTB) selected by calculate_nominal_flexural_strength.

    Args:
        section: WSection providing Z_x, S_x, r_y, J, h_o, r_ts, c.
        yield_stress: F_y (ksi).
        positions: x-coordinates of the moment diagram (inches).
        moments: bending moment at each x (kip-in).
        top_flange_brace_points: x of lateral braces on the top flange.
        bottom_flange_brace_points: x of lateral braces on the bottom flange.
        youngs_modulus: E (ksi), default 29000.
        positive_moment_compresses: flange a positive moment compresses,
            "top" (sagging convention) or "bottom".
    Returns:
        FlexuralSegmentResult per checked segment, sorted by x, with the
        highest demand/capacity segment flagged governing.
    """
    x = np.asarray(positions, dtype=float)
    M = np.asarray(moments, dtype=float)
    order = np.argsort(x)
    x, M = x[order], M[order]
    x_min, x_max = float(x[0]), float(x[-1])
    F_y = float(yield_stress)
    E = float(youngs_modulus)

    top_sign = 1.0 if positive_moment_compresses == "top" else -1.0
    flanges = {
        "top": (top_sign, top_flange_brace_points),
        "bottom": (-top_sign, bottom_flange_brace_points),
    }

    results: list[FlexuralSegmentResult] = []
    for flange, (compress_sign, braces) in flanges.items():
        for start, end, continuous in _brace_segments(braces, x_min, x_max):
            xs, Ms = _slice_moment_segment(x, M, start, end)
            compressing = compress_sign * Ms
            if np.max(compressing) <= zero_atol:
                continue  # flange never in compression here -> no check
            demand = float(np.max(compressing))
            if continuous:
                C_b = 1.0  # irrelevant: L_b = 0 -> plastic
                ltb = calculate_nominal_flexural_strength(section, F_y, 0.0, C_b, E)
            else:
                C_b = calculate_cb_from_moment_diagram(xs, Ms)
                ltb = calculate_nominal_flexural_strength(section, F_y, end - start, C_b, E)
            results.append(FlexuralSegmentResult(flange, start, end, demand, ltb, C_b, continuously_braced=continuous))

    results.sort(key=lambda r: r.x_start)
    if results:
        max(results, key=lambda r: r.demand_capacity_ratio).governing = True
    return results
