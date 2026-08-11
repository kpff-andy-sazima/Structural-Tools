"""AISC 360-22 Section F12 — unsymmetrical shapes other than single angles.

M_n = F_n S_min (Eq. F12-1), where F_n is the lowest of the yield stress and the
lateral-torsional and local buckling stresses obtained from a rational elastic
stability analysis. Those two stresses cannot be derived from section
properties alone, so they must be supplied on the section
(`critical_stress_ltb`, `critical_stress_local`); the corresponding limit
states are reported as not evaluated when they are absent.

Units: kip-inch.
"""

from __future__ import annotations

from ...typing import FloatLike
from .common import Axis, FlexuralStrength, LimitState, missing_note
from .properties import SectionProperties


def check_f12(
    properties: SectionProperties,
    yield_stress: FloatLike,
    axis: Axis = Axis.MAJOR,
) -> FlexuralStrength:
    """Section F12 check at the point of maximum elastic stress.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        axis: axis of bending.
    """
    F_y = float(yield_stress)
    result = FlexuralStrength(
        section_reference="F12", title="Unsymmetrical shapes", axis=axis
    )
    result.warnings.append(
        "F12 requires a rational elastic stability analysis for F_cr; all limit "
        "states are evaluated at the point of maximum elastic stress."
    )

    S_min = properties.section_modulus_x if axis is Axis.MAJOR else properties.section_modulus_y
    if S_min is None:
        result.add(
            LimitState("Y", "Yielding", "F12-1", note=missing_note(["section modulus S_min"]))
        )
        return result
    result.intermediate_values["S_min"] = S_min

    result.add(LimitState("Y", "Yielding", "F12-1 / F12-2", F_y * S_min))

    if properties.critical_stress_ltb is None:
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling",
                "F12-3",
                note=missing_note(["critical_stress_ltb"]),
            )
        )
    else:
        F_cr = min(float(properties.critical_stress_ltb), F_y)
        result.add(LimitState("LTB", "Lateral-torsional buckling", "F12-3", F_cr * S_min))

    if properties.critical_stress_local is None:
        result.add(
            LimitState(
                "LB", "Local buckling", "F12-4", note=missing_note(["critical_stress_local"])
            )
        )
    else:
        F_cr = min(float(properties.critical_stress_local), F_y)
        result.add(LimitState("LB", "Local buckling", "F12-4", F_cr * S_min))
    return result
