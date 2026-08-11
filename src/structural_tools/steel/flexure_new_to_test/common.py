"""Shared types and AISC 360-22 Section F1 general provisions.

Every Chapter F module in this package returns `LimitState` objects collected
into a `FlexuralStrength`; the nominal strength is always the lowest applicable
limit state, per the User Note table in Section F1.

Units: kip, inch (moments kip-in, stresses ksi, lengths in).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

import numpy as np

from ...typing import FloatLike

#: Resistance factor and safety factor for flexure (AISC 360-22 Section F1).
FLEXURE_RESISTANCE_FACTOR = 0.90  # phi_b, LRFD
FLEXURE_SAFETY_FACTOR = 1.67  # Omega_b, ASD


class Axis(Enum):
    """Axis of bending.

    MAJOR: strong axis (x-x) bending.
    MINOR: weak axis (y-y) bending.
    """

    MAJOR = "major"
    MINOR = "minor"


class ShapeGroup(Enum):
    """Cross-section family, used to route to the correct Chapter F section."""

    I_SHAPE = "i_shape"  # W, M, S, HP
    BUILT_UP_I = "built_up_i"  # welded plate girders
    CHANNEL = "channel"  # C, MC
    HSS_RECT = "hss_rect"  # square/rectangular HSS and box sections
    HSS_ROUND = "hss_round"  # round HSS and pipe
    TEE = "tee"  # WT, MT, ST
    DOUBLE_ANGLE = "double_angle"  # 2L
    SINGLE_ANGLE = "single_angle"  # L
    RECT_BAR = "rect_bar"  # rectangular bars and plates
    ROUND_BAR = "round_bar"  # rounds
    UNSYMMETRICAL = "unsymmetrical"  # everything else -> F12


class ElementClass(Enum):
    """Table B4.1b classification of a cross-section element for flexure."""

    COMPACT = "compact"
    NONCOMPACT = "noncompact"
    SLENDER = "slender"


class Bracing(Enum):
    """Lateral bracing condition for a single flange.

    CONTINUOUS: braced along the entire length (e.g. by a floor slab/deck),
        so the flange cannot buckle laterally and develops M_p (L_b = 0).
    UNBRACED: no bracing. L_b = full span.
    TENSION_ONLY: braced flange is the tension flange.
    """

    CONTINUOUS = "continuous"
    UNBRACED = "unbraced"
    TENSION_ONLY = "tension_only"


class StemOrientation(Enum):
    """Tee stem / double-angle web leg state over the unbraced length (F9)."""

    TENSION = "tension"
    COMPRESSION = "compression"


class AngleBending(Enum):
    """Single-angle bending basis (F10.2).

    PRINCIPAL: bending about the major principal (w-w) axis, Eq. F10-4.
    GEOMETRIC_UNRESTRAINED: geometric axis, no lateral-torsional restraint;
        M_y and S_c are taken as 0.80 of the geometric-axis values.
    GEOMETRIC_RESTRAINED: geometric axis, restrained at the point of maximum
        moment only; M_cr is increased 25% and M_y uses the full geometric S.
    """

    PRINCIPAL = "principal"
    GEOMETRIC_UNRESTRAINED = "geometric_unrestrained"
    GEOMETRIC_RESTRAINED = "geometric_restrained"


class LegTip(Enum):
    """State of the angle leg toe over the unbraced length (F10)."""

    COMPRESSION = "compression"
    TENSION = "tension"


# A flange's bracing is either explicit brace-point x-coordinates or a
# Bracing member.
BracePoints = Union[Sequence[FloatLike], Bracing]


@dataclass
class LimitState:
    """One Chapter F limit state.

    A limit state that does not apply carries `nominal_moment = None` and a
    `note` explaining why, so calculation reports can show it greyed out.
    """

    key: str  # "Y" | "CFY" | "LTB" | "FLB" | "WLB" | "SLB" | "LLB" | "TFY" | "RUP"
    name: str
    equation: str  # AISC 360-22 equation reference, e.g. "F2-2"
    nominal_moment: Optional[float] = None  # M_n, kip-in
    note: str = ""
    governs: bool = False

    @property
    def applies(self) -> bool:
        """True when this limit state produced a usable M_n."""
        return self.nominal_moment is not None and np.isfinite(self.nominal_moment)


@dataclass
class FlexuralStrength:
    """Result of a full Chapter F check of one section about one axis.

    Attributes:
        section_reference: governing Chapter F section, e.g. "F4".
        title: the Specification heading for that section.
        axis: axis of bending checked.
        limit_states: every limit state considered, applicable or not.
        intermediate_values: named intermediates (M_p, L_p, L_r, R_pc, ...) in
            kip-inch units, for reporting.
        classification: Table B4.1b results keyed by "flange" / "web".
        warnings: assumptions or out-of-scope notes raised during the check.
    """

    section_reference: str
    title: str
    axis: Axis
    limit_states: list[LimitState] = field(default_factory=list)
    intermediate_values: dict[str, float] = field(default_factory=dict)
    classification: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def applicable(self) -> list[LimitState]:
        """Limit states that produced a usable M_n."""
        return [ls for ls in self.limit_states if ls.applies]

    @property
    def governing(self) -> Optional[LimitState]:
        """Lowest-strength applicable limit state."""
        states = self.applicable
        return min(states, key=lambda ls: ls.nominal_moment) if states else None

    @property
    def nominal_moment(self) -> Optional[float]:
        """M_n, the lowest applicable limit state (kip-in)."""
        governing = self.governing
        return governing.nominal_moment if governing else None

    def add(self, limit_state: LimitState) -> None:
        """Append a limit state and re-flag the governing one."""
        self.limit_states.append(limit_state)
        for state in self.limit_states:
            state.governs = False
        governing = self.governing
        if governing is not None:
            governing.governs = True

    @property
    def design_moment(self) -> Optional[float]:
        """phi_b * M_n (LRFD), kip-in."""
        M_n = self.nominal_moment
        return None if M_n is None else FLEXURE_RESISTANCE_FACTOR * M_n

    @property
    def allowable_moment(self) -> Optional[float]:
        """M_n / Omega_b (ASD), kip-in."""
        M_n = self.nominal_moment
        return None if M_n is None else M_n / FLEXURE_SAFETY_FACTOR


def design_flexural_strength(nominal_moment: FloatLike) -> float:
    """phi_b * M_n with phi_b = 0.90 (LRFD, Section F1)."""
    return FLEXURE_RESISTANCE_FACTOR * float(nominal_moment)


def allowable_flexural_strength(nominal_moment: FloatLike) -> float:
    """M_n / Omega_b with Omega_b = 1.67 (ASD, Section F1)."""
    return float(nominal_moment) / FLEXURE_SAFETY_FACTOR


def calculate_lateral_torsional_buckling_modification_factor(
    max_moment: FloatLike,
    quarter_moment: FloatLike,
    mid_moment: FloatLike,
    three_quarter_moment: FloatLike,
) -> float:
    """LTB modification factor C_b via the quarter-point method (Eq. F1-1).

    Absolute moment values are used internally.

    Args:
        max_moment: M_max, largest moment in the unbraced segment
        quarter_moment: M_A, moment at the quarter point
        mid_moment: M_B, moment at midspan of the segment
        three_quarter_moment: M_C, moment at the three-quarter point
    """
    M_max = abs(float(max_moment))
    M_a = abs(float(quarter_moment))
    M_b = abs(float(mid_moment))
    M_c = abs(float(three_quarter_moment))
    denominator = 2.5 * M_max + 3 * M_a + 4 * M_b + 3 * M_c
    if denominator <= 0:
        return 1.0
    return 12.5 * M_max / denominator


def calculate_cb_from_moment_diagram(
    positions: Sequence[FloatLike],
    moments: Sequence[FloatLike],
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
    return calculate_lateral_torsional_buckling_modification_factor(
        np.max(np.abs(M)), quarter, mid, three_quarter
    )


def interpolate_noncompact(
    plastic_capacity: FloatLike,
    residual_capacity: FloatLike,
    slenderness: FloatLike,
    limiting_compact: FloatLike,
    limiting_noncompact: FloatLike,
) -> float:
    """Linear noncompact interpolation shared by F3-1, F4-13, F6-2, F7-2, F9-14.

    Args:
        plastic_capacity: value at lambda = lambda_p (e.g. M_p, F_y)
        residual_capacity: value at lambda = lambda_r (e.g. 0.7 F_y S_x)
        slenderness: lambda
        limiting_compact: lambda_p
        limiting_noncompact: lambda_r
    """
    top = float(plastic_capacity)
    bottom = float(residual_capacity)
    lam = float(slenderness)
    lam_p = float(limiting_compact)
    lam_r = float(limiting_noncompact)
    if lam_r <= lam_p:
        return bottom
    return top - (top - bottom) * (lam - lam_p) / (lam_r - lam_p)


def classify(
    slenderness: Optional[FloatLike],
    limiting_compact: FloatLike,
    limiting_noncompact: FloatLike,
) -> Optional[ElementClass]:
    """Compact / noncompact / slender per Section B4.1, or None if lambda is unknown."""
    if slenderness is None:
        return None
    lam = float(slenderness)
    if lam <= float(limiting_compact):
        return ElementClass.COMPACT
    if lam <= float(limiting_noncompact):
        return ElementClass.NONCOMPACT
    return ElementClass.SLENDER


def missing(properties, names: Sequence[str]) -> list[str]:
    """Names in `names` that are None or non-finite on `properties`."""
    absent = []
    for name in names:
        value = getattr(properties, name, None)
        if value is None or not np.isfinite(value):
            absent.append(name)
    return absent


def missing_note(names: Sequence[str]) -> str:
    """Standard 'limit state skipped' note for absent section properties."""
    return "requires " + ", ".join(names)
