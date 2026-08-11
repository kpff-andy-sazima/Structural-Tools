"""Chapter F dispatcher — pick the applicable Specification section, run it, and
fold in the Section F13.1 tension-flange rupture check.

Routing follows Table User Note F1.1. The nominal strength returned is the
lowest applicable limit state; limit states that do not apply are retained with
the reason so a calculation report can show them greyed out.

Units: kip-inch.
"""

from __future__ import annotations

from typing import Optional, Union

from ...typing import FloatLike
from .classification import classify_section
from .common import (
    AngleBending,
    Axis,
    ElementClass,
    FlexuralStrength,
    LegTip,
    ShapeGroup,
    StemOrientation,
)
from ..constants import YOUNGS_MODULUS_KSI
from .f2 import check_f2
from .f3 import check_f3
from .f4 import check_f4
from .f5 import check_f5
from .f6 import check_f6
from .f7 import check_f7
from .f8 import check_f8
from .f9 import check_f9
from .f10 import check_f10
from .f11 import check_f11
from .f12 import check_f12
from .f13 import check_tension_flange_rupture
from .properties import SectionProperties

I_SHAPED = (ShapeGroup.I_SHAPE, ShapeGroup.BUILT_UP_I, ShapeGroup.CHANNEL)


def calculate_nominal_flexural_strength(
    section: Union[SectionProperties, "object"],
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    axis: Axis = Axis.MAJOR,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
    tensile_strength: Optional[FloatLike] = None,
    stem: StemOrientation = StemOrientation.TENSION,
    angle_bending: AngleBending = AngleBending.PRINCIPAL,
    leg_tip: LegTip = LegTip.COMPRESSION,
) -> FlexuralStrength:
    """Nominal flexural strength M_n per AISC 360-22 Chapter F.

    Args:
        section: SectionProperties, or a WSection which is adapted automatically.
        yield_stress: F_y (ksi).
        unbraced_length: L_b (in). Use 0.0 for a continuously braced flange.
        axis: axis of bending.
        ltb_modification_factor: C_b (Eq. F1-1). F9 ignores it and F10 caps it
            at 1.5, per those sections.
        youngs_modulus: E (ksi).
        tensile_strength: F_u (ksi), only needed for the F13.1 rupture check.
        stem: tee stem / double-angle web leg state over L_b (F9).
        angle_bending: single-angle bending basis (F10.2).
        leg_tip: single-angle leg toe state (F10).

    Returns:
        FlexuralStrength whose `nominal_moment` is the lowest applicable limit
        state and whose `governing` names the equation that controls.
    """
    properties = _coerce(section).derived(yield_stress, youngs_modulus)
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    L_b = float(unbraced_length)
    C_b = float(ltb_modification_factor)

    plastic_moment = (
        min(F_y * properties.plastic_modulus_x, 1.6 * F_y * properties.section_modulus_x)
        if properties.plastic_modulus_x and properties.section_modulus_x
        else None
    )
    compression_yield_moment = (
        F_y * properties.section_modulus_compression
        if properties.section_modulus_compression
        else None
    )
    checks = classify_section(properties, F_y, E, plastic_moment, compression_yield_moment)
    flange = checks.get("flange")
    web = checks.get("web")

    group = properties.group
    if group in I_SHAPED:
        if axis is Axis.MINOR:
            result = check_f6(properties, F_y, flange, E)
        else:
            result = _dispatch_i_shape(properties, F_y, L_b, flange, web, C_b, E)
    elif group is ShapeGroup.HSS_RECT:
        result = check_f7(properties, F_y, L_b, flange, web, axis, C_b, E)
    elif group is ShapeGroup.HSS_ROUND:
        result = check_f8(properties, F_y, flange, E)
    elif group in (ShapeGroup.TEE, ShapeGroup.DOUBLE_ANGLE):
        result = check_f9(properties, F_y, L_b, flange, web, stem, E)
    elif group is ShapeGroup.SINGLE_ANGLE:
        result = check_f10(properties, F_y, L_b, flange, angle_bending, leg_tip, C_b, E)
    elif group in (ShapeGroup.RECT_BAR, ShapeGroup.ROUND_BAR):
        result = check_f11(properties, F_y, L_b, axis, C_b, E)
    else:
        result = check_f12(properties, F_y, axis)

    result.classification = checks
    result.axis = axis
    if tensile_strength is not None:
        rupture = check_tension_flange_rupture(properties, F_y, tensile_strength, axis)
        if rupture is not None:
            result.add(rupture)
    return result


def _dispatch_i_shape(
    properties: SectionProperties,
    yield_stress: float,
    unbraced_length: float,
    flange,
    web,
    ltb_modification_factor: float,
    youngs_modulus: float,
) -> FlexuralStrength:
    """Route a major-axis I-shape or channel to F2, F3, F4 or F5."""
    web_class = web.classification if web and web.known else ElementClass.COMPACT
    flange_class = flange.classification if flange and flange.known else ElementClass.COMPACT

    if web_class is ElementClass.SLENDER:
        return check_f5(
            properties, yield_stress, unbraced_length, flange, ltb_modification_factor, youngs_modulus
        )
    if web_class is ElementClass.NONCOMPACT or properties.singly_symmetric:
        return check_f4(
            properties,
            yield_stress,
            unbraced_length,
            flange,
            web,
            ltb_modification_factor,
            youngs_modulus,
        )
    if flange_class is ElementClass.COMPACT:
        return check_f2(
            properties, yield_stress, unbraced_length, ltb_modification_factor, youngs_modulus
        )

    result = check_f3(
        properties, yield_stress, unbraced_length, flange, ltb_modification_factor, youngs_modulus
    )
    if properties.group is ShapeGroup.CHANNEL:
        result.section_reference = "F2 + F3"
        result.title = (
            "Channel with a noncompact or slender flange — F2.2 lateral-torsional "
            "buckling with an F3-form flange local buckling check"
        )
        result.warnings.append(
            "F3 is written for doubly symmetric I-shapes. The flange local buckling "
            "check applied to this channel follows the F3 form; confirm the treatment "
            "is acceptable for the project."
        )
    return result


def _coerce(section) -> SectionProperties:
    """Return `section` as SectionProperties, adapting a WSection if needed."""
    if isinstance(section, SectionProperties):
        return section
    return SectionProperties.from_wsection(section)
