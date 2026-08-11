"""AISC 360-22 Section F6 — I-shaped members and channels bent about their
minor axis. Limit states: yielding, flange local buckling. LTB cannot occur.

Note the F6 flange slenderness uses lambda = b/t_f, where b is half the flange
width for I-shapes and the full flange width for channels — the same lambda as
Table B4.1b, so the classification from `classify_section` is reused directly.

Units: kip-inch.
"""

from __future__ import annotations

from typing import Optional

from ...typing import FloatLike
from .classification import SlendernessCheck
from .common import (
    Axis,
    ElementClass,
    FlexuralStrength,
    LimitState,
    interpolate_noncompact,
    missing,
    missing_note,
)
from ..constants import YOUNGS_MODULUS_KSI
from .properties import SectionProperties


def calculate_minor_axis_plastic_moment(
    yield_stress: FloatLike,
    plastic_section_modulus_y_axis: FloatLike,
    section_modulus_y_axis: FloatLike,
) -> float:
    """Minor-axis plastic moment M_p = F_y Z_y <= 1.6 F_y S_y (Eq. F6-1)."""
    F_y = float(yield_stress)
    return min(F_y * float(plastic_section_modulus_y_axis), 1.6 * F_y * float(section_modulus_y_axis))


def calculate_slender_flange_stress(
    flange_slenderness: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Critical stress for a slender minor-axis flange, F_cr (Eq. F6-4)."""
    return 0.69 * float(youngs_modulus) / float(flange_slenderness) ** 2


def check_f6(
    properties: SectionProperties,
    yield_stress: FloatLike,
    flange: Optional[SlendernessCheck],
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F6 check.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        flange: Table B4.1b check for the flange.
        youngs_modulus: E (ksi).
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)

    result = FlexuralStrength(
        section_reference="F6",
        title="I-shaped members and channels bent about their minor axis",
        axis=Axis.MINOR,
    )
    absent = missing(properties, ["plastic_modulus_y", "section_modulus_y"])
    if absent:
        result.add(LimitState("Y", "Yielding", "F6-1", note=missing_note(absent)))
        return result

    S_y = properties.section_modulus_y
    M_p = calculate_minor_axis_plastic_moment(F_y, properties.plastic_modulus_y, S_y)
    result.intermediate_values["M_p"] = M_p
    result.add(LimitState("Y", "Yielding (plastic moment)", "F6-1", M_p))
    result.add(
        LimitState(
            "LTB",
            "Lateral-torsional buckling",
            "—",
            note="minor-axis bending — the limit state does not apply",
        )
    )

    if flange is None or not flange.known:
        result.add(
            LimitState(
                "FLB", "Flange local buckling", "F6-2", note=missing_note(["flange_slenderness"])
            )
        )
    elif flange.classification is ElementClass.COMPACT:
        result.add(
            LimitState(
                "FLB",
                "Flange local buckling",
                "F6-2",
                note="compact flange — the limit state does not apply",
            )
        )
    elif flange.classification is ElementClass.NONCOMPACT:
        M_n = interpolate_noncompact(
            M_p,
            0.7 * F_y * S_y,
            flange.slenderness,
            flange.limiting_compact,
            flange.limiting_noncompact,
        )  # F6-2
        result.add(LimitState("FLB", "Flange local buckling (noncompact)", "F6-2", M_n))
    else:
        F_cr = calculate_slender_flange_stress(flange.slenderness, E)
        result.intermediate_values["F_cr"] = F_cr
        result.add(
            LimitState("FLB", "Flange local buckling (slender)", "F6-3 / F6-4", F_cr * S_y)
        )
    return result
