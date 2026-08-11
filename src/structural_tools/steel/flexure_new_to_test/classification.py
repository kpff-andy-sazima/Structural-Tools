"""Table B4.1b element slenderness classification for flexure.

Chapter F routes on these results (compact / noncompact / slender webs and
flanges), so the table lives with the flexure code rather than in a general
B4 module. Case numbers are deliberately not cited: they have shifted between
Specification editions, so each check names the element instead.

Units: kip, inch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

from ...typing import FloatLike
from .common import ElementClass, ShapeGroup, classify
from .properties import SectionProperties


@dataclass
class SlendernessCheck:
    """One Table B4.1b width-to-thickness check.

    Attributes:
        element: "flange" or "web".
        symbol: the ratio evaluated, e.g. "b_f/2t_f".
        slenderness: lambda.
        limiting_compact: lambda_p.
        limiting_noncompact: lambda_r.
        classification: resulting ElementClass, or None if lambda is unknown.
        description: element description as worded in Table B4.1b.
    """

    element: str
    symbol: str
    slenderness: Optional[float]
    limiting_compact: float
    limiting_noncompact: float
    classification: Optional[ElementClass]
    description: str

    @property
    def known(self) -> bool:
        """True when the check produced a classification."""
        return self.classification is not None


def classify_section(
    properties: SectionProperties,
    yield_stress: FloatLike,
    youngs_modulus: FloatLike,
    plastic_moment: Optional[FloatLike] = None,
    compression_yield_moment: Optional[FloatLike] = None,
) -> dict[str, SlendernessCheck]:
    """Classify the flange and web of `properties` per Table B4.1b.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        youngs_modulus: E (ksi).
        plastic_moment: M_p, only needed for singly symmetric I-shaped webs.
        compression_yield_moment: M_yc, only needed for singly symmetric webs.

    Returns:
        Mapping with optional "flange" and "web" SlendernessCheck entries.
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    root = sqrt(E / F_y)
    group = properties.group
    checks: dict[str, SlendernessCheck] = {}

    if group in (ShapeGroup.I_SHAPE, ShapeGroup.BUILT_UP_I, ShapeGroup.CHANNEL, ShapeGroup.TEE):
        limiting_compact = 0.38 * root
        limiting_noncompact = 1.0 * root
        description = "flanges of rolled I-shaped sections, channels and tees"
        welded = properties.welded or group is ShapeGroup.BUILT_UP_I
        if welded and properties.flange_buckling_coefficient is not None:
            F_L = properties.limiting_flange_stress or 0.7 * F_y
            limiting_noncompact = 0.95 * sqrt(properties.flange_buckling_coefficient * E / F_L)
            description = "flanges of built-up I-shaped sections"
        symbol = "b/t" if group is ShapeGroup.CHANNEL else "b_f/2t_f"
        checks["flange"] = _check(
            "flange",
            symbol,
            properties.flange_slenderness,
            limiting_compact,
            limiting_noncompact,
            description,
        )

    if group in (ShapeGroup.I_SHAPE, ShapeGroup.BUILT_UP_I, ShapeGroup.CHANNEL):
        limiting_noncompact = 5.70 * root
        limiting_compact = 3.76 * root
        description = "webs of doubly symmetric I-shaped sections and channels"
        if (
            properties.singly_symmetric
            and plastic_moment
            and compression_yield_moment
            and properties.web_depth_plastic
            and properties.web_depth_compression
        ):
            ratio = float(plastic_moment) / float(compression_yield_moment)
            denominator = (0.54 * ratio - 0.09) ** 2
            limiting_compact = min(
                (properties.web_depth_compression / properties.web_depth_plastic)
                * root
                / denominator,
                limiting_noncompact,
            )
            description = "webs of singly symmetric I-shaped sections"
        checks["web"] = _check(
            "web",
            "h/t_w",
            properties.web_slenderness,
            limiting_compact,
            limiting_noncompact,
            description,
        )

    if group is ShapeGroup.TEE:
        checks["web"] = _check(
            "web",
            "d/t_w",
            properties.web_slenderness,
            0.84 * root,
            1.52 * root,
            "stems of tees",
        )

    if group is ShapeGroup.HSS_RECT:
        checks["flange"] = _check(
            "flange",
            "b/t",
            properties.flange_slenderness,
            1.12 * root,
            1.40 * root,
            "flanges of rectangular HSS and box sections",
        )
        checks["web"] = _check(
            "web",
            "h/t",
            properties.web_slenderness,
            2.42 * root,
            5.70 * root,
            "webs of rectangular HSS and box sections",
        )

    if group is ShapeGroup.HSS_ROUND:
        checks["flange"] = _check(
            "flange",
            "D/t",
            properties.diameter_slenderness,
            0.07 * E / F_y,
            0.31 * E / F_y,
            "round HSS",
        )

    if group in (ShapeGroup.SINGLE_ANGLE, ShapeGroup.DOUBLE_ANGLE):
        checks["flange"] = _check(
            "flange",
            "b/t",
            properties.flange_slenderness,
            0.54 * root,
            0.91 * root,
            "legs of single angles and double angles with separators",
        )

    return checks


def _check(
    element: str,
    symbol: str,
    slenderness: Optional[float],
    limiting_compact: float,
    limiting_noncompact: float,
    description: str,
) -> SlendernessCheck:
    """Assemble one SlendernessCheck."""
    return SlendernessCheck(
        element=element,
        symbol=symbol,
        slenderness=None if slenderness is None else float(slenderness),
        limiting_compact=limiting_compact,
        limiting_noncompact=limiting_noncompact,
        classification=classify(slenderness, limiting_compact, limiting_noncompact),
        description=description,
    )
