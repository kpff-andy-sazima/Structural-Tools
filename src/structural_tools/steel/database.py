"""Look up AISC sections from steelpy and adapt them to WSection.

steelpy is the live AISC Shapes Database (kip-inch). This module is the only
place that depends on steelpy's API, so the rest of structural_tools depends
only on WSection.
"""

import re

from steelpy import aisc

from .section import WSection

_COLLECTION = {
    "W": "W_shapes",
    "M": "M_shapes",
    "S": "S_shapes",
    "HP": "HP_shapes",
    "C": "C_shapes",
    "MC": "MC_shapes",
}
_SECTION_TYPE = {
    "W": "i_shape",
    "M": "i_shape",
    "S": "i_shape",
    "HP": "i_shape",
    "C": "channel",
    "MC": "channel",
}


def _prefix(label: str) -> str:
    match = re.match(r"[A-Z]+", label)
    return match.group(0) if match else ""


def get_section(name: str) -> WSection:
    """Return a WSection for an AISC manual label, e.g. "W18X50", "C15X50".

    Raises KeyError for shapes outside F2 scope (WT/MT/ST tees, L angles,
    HSS, PIPE) or labels not found in the steelpy database.
    """
    label = name.upper().strip()
    prefix = _prefix(label)
    if prefix not in _COLLECTION:
        raise KeyError(f"{name!r}: prefix {prefix!r} is not an F2 shape (supported: W, M, S, HP, C, MC).")

    collection = getattr(aisc, _COLLECTION[prefix])
    try:
        prof = getattr(collection, label)  # e.g. aisc.W_shapes.W18X50
    except AttributeError as exc:
        raise KeyError(f"{name!r} not found in AISC database.") from exc

    return WSection(
        plastic_section_modulus=prof.Zx,
        section_modulus_x_axis=prof.Sx,
        second_moment_of_area_y_axis=prof.Iy,
        warping_constant=prof.Cw,
        radius_of_gyration_y_axis=prof.ry,
        torsional_constant=prof.J,
        flange_centroid_distance=prof.ho,
        name=label,
        section_type=_SECTION_TYPE[prefix],
    )
