"""Beam-level flexure check driven directly by a moment diagram.

Each flange is broken into unbraced segments by its own brace points, the
compression flange is chosen from the sign of the moment, and every segment
where that flange sees compression is checked through the Chapter F dispatcher
with L_b = segment length and C_b from the quarter-point method.

Units: kip-inch.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from ...typing import FloatLike
from .common import (
    Axis,
    BracePoints,
    Bracing,
    FlexuralStrength,
    calculate_cb_from_moment_diagram,
)
from ..constants import YOUNGS_MODULUS_KSI
from .dispatch import calculate_nominal_flexural_strength
from .properties import SectionProperties


@dataclass
class FlexuralSegmentResult:
    """One unbraced-segment flexure check on a single compression flange.

    Moments in kip-in, lengths in inches.
    """

    flange: str  # "top" | "bottom"
    x_start: float
    x_end: float
    moment_demand: float  # M_u causing compression in this flange
    strength: FlexuralStrength
    ltb_modification_factor: float  # C_b
    continuously_braced: bool = False
    governing: bool = False

    @property
    def unbraced_length(self) -> float:
        """L_b used for this segment (0.0 when continuously braced)."""
        return 0.0 if self.continuously_braced else self.x_end - self.x_start

    @property
    def nominal_moment(self) -> Optional[float]:
        """M_n for this segment."""
        return self.strength.nominal_moment

    @property
    def demand_capacity_ratio(self) -> float:
        """M_u / M_n (apply phi = 0.90 or Omega = 1.67 at the call site)."""
        M_n = self.nominal_moment
        return float("inf") if not M_n else self.moment_demand / M_n

    @property
    def governing_equation(self) -> str:
        """AISC equation reference for the limit state that controls M_n."""
        governing = self.strength.governing
        return governing.equation if governing else ""

    @property
    def governing_limit_state(self) -> str:
        """Name of the limit state that controls M_n."""
        governing = self.strength.governing
        return governing.name if governing else ""


def evaluate_beam_flexure(
    section: Union[SectionProperties, "object"],
    yield_stress: FloatLike,
    positions: Sequence[FloatLike],
    moments: Sequence[FloatLike],
    top_flange_brace_points: BracePoints,
    bottom_flange_brace_points: BracePoints,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
    positive_moment_compresses: str = "bottom",
    zero_atol: float = 1e-6,
    **chapter_f_options,
) -> list[FlexuralSegmentResult]:
    """Full major-axis flexure check of a beam directly from a moment diagram.

    The compression flange is chosen automatically from the sign of the moment.
    Each flange is broken into unbraced segments by its own brace points (member
    ends added automatically); segments where that flange sees compression are
    checked with L_b = segment length, C_b from the quarter-point method over
    the segment, and the governing Chapter F section selected by
    `calculate_nominal_flexural_strength`.

    Args:
        section: SectionProperties or WSection.
        yield_stress: F_y (ksi).
        positions: x-coordinates of the moment diagram (inches).
        moments: bending moment at each x (kip-in).
        top_flange_brace_points: x of lateral braces on the top flange, or a
            Bracing member.
        bottom_flange_brace_points: same for the bottom flange.
        youngs_modulus: E (ksi), default 29000.
        positive_moment_compresses: flange a positive moment compresses,
            "top" (sagging convention) or "bottom".
        zero_atol: compression below this magnitude is treated as none.
        **chapter_f_options: forwarded to `calculate_nominal_flexural_strength`
            (e.g. tensile_strength, stem, angle_bending, leg_tip).

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
                L_b = 0.0
            else:
                C_b = calculate_cb_from_moment_diagram(xs, Ms)
                L_b = end - start
            strength = calculate_nominal_flexural_strength(
                section,
                F_y,
                L_b,
                axis=Axis.MAJOR,
                ltb_modification_factor=C_b,
                youngs_modulus=E,
                **chapter_f_options,
            )
            results.append(
                FlexuralSegmentResult(
                    flange, start, end, demand, strength, C_b, continuously_braced=continuous
                )
            )

    results.sort(key=lambda r: r.x_start)
    if results:
        max(results, key=lambda r: r.demand_capacity_ratio).governing = True
    return results


def flexure_results_to_dataframe(
    results: Sequence[FlexuralSegmentResult],
    moment_divisor: float = 12.0,
    length_divisor: float = 12.0,
) -> pd.DataFrame:
    """Tabulate segment results for a calculation report.

    Args:
        results: output of `evaluate_beam_flexure`.
        moment_divisor: 12.0 converts kip-in to kip-ft; use 1.0 to stay in kip-in.
        length_divisor: 12.0 converts inches to feet.
    """
    rows = []
    for result in results:
        rows.append(
            {
                "flange": result.flange,
                "x_start": result.x_start / length_divisor,
                "x_end": result.x_end / length_divisor,
                "L_b": result.unbraced_length / length_divisor,
                "C_b": result.ltb_modification_factor,
                "M_u": result.moment_demand / moment_divisor,
                "M_n": (result.nominal_moment or float("nan")) / moment_divisor,
                "equation": result.governing_equation,
                "limit state": result.governing_limit_state,
                "M_u/M_n": result.demand_capacity_ratio,
                "governs": result.governing,
            }
        )
    return pd.DataFrame(rows)


def _brace_segments(
    braces: BracePoints, x_min: float, x_max: float
) -> Iterator[tuple[float, float, bool]]:
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
