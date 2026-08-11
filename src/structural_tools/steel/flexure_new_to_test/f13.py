"""AISC 360-22 Section F13 — proportions of beams and girders.

F13.1 (tensile rupture of the flange) reduces M_n and is folded into the
dispatcher's limit-state list. F13.2 and F13.3 are proportioning limits rather
than strength equations, so they are returned as pass/fail checks for the
calculation report.

Units: kip-inch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from ...typing import FloatLike
from .common import Axis, LimitState
from ..constants import YOUNGS_MODULUS_KSI
from .properties import SectionProperties


def calculate_hole_reduction_coefficient(
    yield_stress: FloatLike,
    tensile_strength: FloatLike,
) -> float:
    """Y_t: 1.0 when F_y/F_u <= 0.8, otherwise 1.1 (Section F13.1)."""
    return 1.0 if float(yield_stress) / float(tensile_strength) <= 0.8 else 1.1


def check_tension_flange_rupture(
    properties: SectionProperties,
    yield_stress: FloatLike,
    tensile_strength: FloatLike,
    axis: Axis = Axis.MAJOR,
) -> Optional[LimitState]:
    """Tensile rupture of the tension flange (Section F13.1, Eq. F13-1).

    Returns None when A_fg or A_fn is absent (no holes to consider).

    Args:
        properties: derived SectionProperties carrying A_fg and A_fn.
        yield_stress: F_y (ksi).
        tensile_strength: F_u (ksi).
        axis: axis of bending.
    """
    A_fg = properties.gross_flange_area
    A_fn = properties.net_flange_area
    if not A_fg or not A_fn:
        return None
    S = (
        properties.section_modulus_tension or properties.section_modulus_x
        if axis is Axis.MAJOR
        else properties.section_modulus_y
    )
    if S is None:
        return None

    F_y = float(yield_stress)
    F_u = float(tensile_strength)
    Y_t = calculate_hole_reduction_coefficient(F_y, F_u)
    if F_u * A_fn >= Y_t * F_y * A_fg:
        return LimitState(
            "RUP",
            "Tension flange rupture",
            "F13-1",
            note=f"F_u A_fn >= Y_t F_y A_fg with Y_t = {Y_t:.1f} — no reduction required",
        )
    return LimitState(
        "RUP",
        "Tension flange rupture",
        "F13-1",
        F_u * A_fn / A_fg * S,
        note=f"Y_t = {Y_t:.1f}",
    )


@dataclass
class ProportioningCheck:
    """One Section F13.2 proportioning limit.

    Attributes:
        name: what is being limited.
        value: the computed quantity.
        limit: the permitted value.
        satisfied: True when the limit is met.
        reference: Specification reference.
    """

    name: str
    value: float
    limit: float
    satisfied: bool
    reference: str


def check_proportioning_limits(
    properties: SectionProperties,
    yield_stress: FloatLike,
    stiffener_spacing_ratio: Optional[FloatLike] = None,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> list[ProportioningCheck]:
    """Section F13.2 proportioning limits for built-up I-shaped members.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        stiffener_spacing_ratio: a/h, clear stiffener spacing over web depth.
            None is treated as an unstiffened web (a/h > 1.5).
        youngs_modulus: E (ksi).

    Returns:
        Every limit that could be evaluated, in Specification order.
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    checks: list[ProportioningCheck] = []

    if properties.second_moment_compression_flange and properties.second_moment_y:
        ratio = properties.second_moment_compression_flange / properties.second_moment_y
        checks.append(
            ProportioningCheck(
                "I_yc / I_y",
                ratio,
                0.9,
                0.1 <= ratio <= 0.9,
                "F13.2",
            )
        )

    if properties.web_slenderness is not None:
        stiffened = stiffener_spacing_ratio is not None and float(stiffener_spacing_ratio) <= 1.5
        limit = 12.0 * sqrt(E / F_y) if stiffened else 0.40 * E / F_y
        checks.append(
            ProportioningCheck(
                "h/t_w",
                properties.web_slenderness,
                limit,
                properties.web_slenderness <= limit,
                "F13.2",
            )
        )
    return checks
