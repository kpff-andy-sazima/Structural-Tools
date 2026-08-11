"""AISC 360-22 Section F8 — round HSS with D/t < 0.45E/F_y.
Limit states: yielding, local buckling. LTB does not apply.

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
    missing,
    missing_note,
)
from ..constants import YOUNGS_MODULUS_KSI
from .properties import SectionProperties


def calculate_noncompact_local_buckling_moment(
    yield_stress: FloatLike,
    diameter_slenderness: FloatLike,
    section_modulus: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Local buckling of a noncompact round HSS wall (Eq. F8-2)."""
    return (0.021 * float(youngs_modulus) / float(diameter_slenderness) + float(yield_stress)) * float(
        section_modulus
    )


def calculate_slender_local_buckling_stress(
    diameter_slenderness: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Critical stress for a slender round HSS wall, F_cr (Eq. F8-4)."""
    return 0.33 * float(youngs_modulus) / float(diameter_slenderness)


def check_f8(
    properties: SectionProperties,
    yield_stress: FloatLike,
    wall: Optional[SlendernessCheck],
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F8 check.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        wall: Table B4.1b check on D/t.
        youngs_modulus: E (ksi).
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)

    result = FlexuralStrength(section_reference="F8", title="Round HSS", axis=Axis.MAJOR)
    absent = missing(properties, ["plastic_modulus_x", "section_modulus_x", "diameter_slenderness"])
    if absent:
        result.add(LimitState("Y", "Yielding", "F8-1", note=missing_note(absent)))
        return result

    S = properties.section_modulus_x
    slenderness = properties.diameter_slenderness
    scope_limit = 0.45 * E / F_y
    result.intermediate_values.update(
        {"M_p": F_y * properties.plastic_modulus_x, "D_t_limit": scope_limit}
    )
    if slenderness >= scope_limit:
        result.warnings.append(
            f"D/t = {slenderness:.1f} exceeds the F8 scope limit 0.45E/F_y = {scope_limit:.1f}."
        )

    result.add(
        LimitState("Y", "Yielding (plastic moment)", "F8-1", F_y * properties.plastic_modulus_x)
    )
    result.add(
        LimitState(
            "LTB",
            "Lateral-torsional buckling",
            "—",
            note="round HSS — the limit state does not apply",
        )
    )

    if wall is None or not wall.known:
        result.add(LimitState("LB", "Local buckling", "F8-2", note=missing_note(["D/t"])))
    elif wall.classification is ElementClass.COMPACT:
        result.add(
            LimitState(
                "LB",
                "Local buckling",
                "F8-2",
                note="compact wall — the limit state does not apply",
            )
        )
    elif wall.classification is ElementClass.NONCOMPACT:
        M_n = calculate_noncompact_local_buckling_moment(F_y, slenderness, S, E)
        result.add(LimitState("LB", "Local buckling (noncompact)", "F8-2", M_n))
    else:
        F_cr = calculate_slender_local_buckling_stress(slenderness, E)
        result.intermediate_values["F_cr"] = F_cr
        result.add(LimitState("LB", "Local buckling (slender)", "F8-3 / F8-4", F_cr * S))
    return result
