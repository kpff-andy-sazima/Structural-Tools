"""Steel flexural design per AISC 360-22 Chapter F.

Coverage: F1 through F13. Pure equation functions map 1:1 to the Specification
and take plain floats; each `check_f*` driver assembles the limit states for one
Specification section; `calculate_nominal_flexural_strength` routes a section to
the right one per Table User Note F1.1 and returns the lowest applicable limit
state.

Units: kip-inch throughout (stresses ksi, lengths in).

Typical use::

    from structural_tools.steel.flexure import (
        Axis, SectionProperties, ShapeGroup, calculate_nominal_flexural_strength,
    )

    properties = SectionProperties.from_wsection(section, group=ShapeGroup.I_SHAPE)
    result = calculate_nominal_flexural_strength(properties, 50.0, 20 * 12, Axis.MAJOR, 1.14)
    result.nominal_moment, result.governing.equation
"""

from .beam import (
    FlexuralSegmentResult,
    evaluate_beam_flexure,
    flexure_results_to_dataframe,
)
from .classification import SlendernessCheck, classify_section
from .common import (
    FLEXURE_RESISTANCE_FACTOR,
    FLEXURE_SAFETY_FACTOR,
    AngleBending,
    Axis,
    BracePoints,
    Bracing,
    ElementClass,
    FlexuralStrength,
    LegTip,
    LimitState,
    ShapeGroup,
    StemOrientation,
    allowable_flexural_strength,
    calculate_cb_from_moment_diagram,
    calculate_lateral_torsional_buckling_modification_factor,
    design_flexural_strength,
)
from .dispatch import calculate_nominal_flexural_strength
from .f2 import (
    calculate_effective_radius_of_gyration,
    calculate_elastic_lateral_torsional_buckling_stress,
    calculate_inelastic_lateral_torsional_buckling_moment,
    calculate_limiting_inelastic_length,
    calculate_limiting_yield_length,
    calculate_plastic_moment,
    calculate_torsional_coefficient,
    check_f2,
)
from .f3 import check_f3
from .f4 import calculate_web_plastification_factor, check_f4
from .f5 import calculate_bending_strength_reduction_factor, check_f5
from .f6 import check_f6
from .f7 import calculate_effective_width, check_f7
from .f8 import check_f8
from .f9 import check_f9
from .f10 import check_f10
from .f11 import check_f11
from .f12 import check_f12
from .f13 import (
    ProportioningCheck,
    check_proportioning_limits,
    check_tension_flange_rupture,
)
from .properties import SectionProperties

__all__ = [
    # types and enums
    "AngleBending",
    "Axis",
    "BracePoints",
    "Bracing",
    "ElementClass",
    "FlexuralSegmentResult",
    "FlexuralStrength",
    "LegTip",
    "LimitState",
    "ProportioningCheck",
    "SectionProperties",
    "ShapeGroup",
    "SlendernessCheck",
    "StemOrientation",
    # constants
    "FLEXURE_RESISTANCE_FACTOR",
    "FLEXURE_SAFETY_FACTOR",
    # F1 and general
    "allowable_flexural_strength",
    "calculate_cb_from_moment_diagram",
    "calculate_lateral_torsional_buckling_modification_factor",
    "classify_section",
    "design_flexural_strength",
    # dispatcher and beam-level driver
    "calculate_nominal_flexural_strength",
    "evaluate_beam_flexure",
    "flexure_results_to_dataframe",
    # per-section drivers
    "check_f2",
    "check_f3",
    "check_f4",
    "check_f5",
    "check_f6",
    "check_f7",
    "check_f8",
    "check_f9",
    "check_f10",
    "check_f11",
    "check_f12",
    "check_proportioning_limits",
    "check_tension_flange_rupture",
    # frequently used equation functions
    "calculate_bending_strength_reduction_factor",
    "calculate_effective_radius_of_gyration",
    "calculate_effective_width",
    "calculate_elastic_lateral_torsional_buckling_stress",
    "calculate_inelastic_lateral_torsional_buckling_moment",
    "calculate_limiting_inelastic_length",
    "calculate_limiting_yield_length",
    "calculate_plastic_moment",
    "calculate_torsional_coefficient",
    "calculate_web_plastification_factor",
]
