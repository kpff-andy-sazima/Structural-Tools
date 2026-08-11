"""AISC 360-22 Section F11 — rectangular bars and rounds.
Limit states: yielding, lateral-torsional buckling.

Note the yielding caps differ: 1.5 F_y S_x for rectangular bars (Eq. F11-1) and
1.6 F_y S_x for rounds (Eq. F11-2).

Units: kip-inch.
"""

from __future__ import annotations

from ...typing import FloatLike
from .common import Axis, FlexuralStrength, LimitState, ShapeGroup, missing_note
from ..constants import YOUNGS_MODULUS_KSI
from .properties import SectionProperties


def calculate_plastic_moment(
    yield_stress: FloatLike,
    plastic_section_modulus: FloatLike,
    section_modulus: FloatLike,
    round_bar: bool = False,
) -> float:
    """M_p for a bar (Eq. F11-1 for rectangular bars, Eq. F11-2 for rounds)."""
    F_y = float(yield_stress)
    cap = 1.6 if round_bar else 1.5
    return min(F_y * float(plastic_section_modulus), cap * F_y * float(section_modulus))


def calculate_slenderness_parameter(
    unbraced_length: FloatLike,
    bar_depth: FloatLike,
    bar_width: FloatLike,
) -> float:
    """L_b d / t^2, the F11.2 lateral-torsional buckling parameter."""
    return float(unbraced_length) * float(bar_depth) / float(bar_width) ** 2


def calculate_inelastic_moment(
    slenderness_parameter: FloatLike,
    yield_moment: FloatLike,
    plastic_moment: FloatLike,
    yield_stress: FloatLike,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Inelastic LTB of a rectangular bar (Eq. F11-3), capped at M_p."""
    parameter = float(slenderness_parameter)
    M_y = float(yield_moment)
    M_p = float(plastic_moment)
    C_b = float(ltb_modification_factor)
    ratio = float(yield_stress) / float(youngs_modulus)
    return min(C_b * (1.52 - 0.274 * parameter * ratio) * M_y, M_p)


def calculate_elastic_stress(
    slenderness_parameter: FloatLike,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Critical stress for elastic LTB of a rectangular bar, F_cr (Eq. F11-5)."""
    return (
        1.9
        * float(youngs_modulus)
        * float(ltb_modification_factor)
        / float(slenderness_parameter)
    )


def check_f11(
    properties: SectionProperties,
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    axis: Axis = Axis.MAJOR,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F11 check.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        unbraced_length: L_b (in).
        axis: axis of bending.
        ltb_modification_factor: C_b (Eq. F1-1).
        youngs_modulus: E (ksi).
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    C_b = float(ltb_modification_factor)
    L_b = float(unbraced_length)
    major = axis is Axis.MAJOR
    round_bar = properties.group is ShapeGroup.ROUND_BAR

    result = FlexuralStrength(
        section_reference="F11", title="Rectangular bars and rounds", axis=axis
    )
    Z = properties.plastic_modulus_x if major else properties.plastic_modulus_y
    S = properties.section_modulus_x if major else properties.section_modulus_y
    if Z is None or S is None:
        result.add(
            LimitState(
                "Y",
                "Yielding",
                "F11-1",
                note=missing_note(["plastic modulus Z", "section modulus S"]),
            )
        )
        return result

    M_p = calculate_plastic_moment(F_y, Z, S, round_bar)
    result.intermediate_values.update({"M_p": M_p, "M_y": F_y * S})
    result.add(
        LimitState("Y", "Yielding", "F11-2" if round_bar else "F11-1", M_p)
    )

    if round_bar or not major:
        reason = (
            "rounds — the limit state does not apply"
            if round_bar
            else "rectangular bar bent about its minor axis — the limit state does not apply"
        )
        result.add(LimitState("LTB", "Lateral-torsional buckling", "—", note=reason))
        return result

    if properties.bar_depth is None or properties.bar_width is None:
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling",
                "F11-3",
                note=missing_note(["bar_depth", "bar_width"]),
            )
        )
        return result

    parameter = calculate_slenderness_parameter(L_b, properties.bar_depth, properties.bar_width)
    lower_limit = 0.08 * E / F_y
    upper_limit = 1.9 * E / F_y
    result.intermediate_values.update(
        {"L_b_d_over_t2": parameter, "limit_yield": lower_limit, "limit_elastic": upper_limit}
    )

    if parameter <= lower_limit:
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling",
                "F11-3",
                note="L_b d/t^2 <= 0.08E/F_y — the limit state does not apply",
            )
        )
    elif parameter <= upper_limit:
        M_n = calculate_inelastic_moment(parameter, F_y * S, M_p, F_y, C_b, E)
        result.add(LimitState("LTB", "Lateral-torsional buckling (inelastic)", "F11-3", M_n))
    else:
        F_cr = calculate_elastic_stress(parameter, C_b, E)
        result.intermediate_values["F_cr"] = F_cr
        result.add(
            LimitState(
                "LTB", "Lateral-torsional buckling (elastic)", "F11-4 / F11-5", min(F_cr * S, M_p)
            )
        )
    return result
