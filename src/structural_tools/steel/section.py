"""Structural steel section definitions and derived geometric properties.

A WSection bundles the fundamental properties published in the AISC Shapes
Database and exposes derived quantities (r_ts, c) as computed properties, so
they are calculated once, in one place, and stay consistent across every
calculation and report that consumes the section.

Units: kip-inch (S/Z in in^3, I in in^4, Cw in in^6, lengths in in).
"""

from dataclasses import dataclass
from typing import Literal

from .flexure import (
    calculate_effective_radius_of_gyration,
    calculate_torsional_coefficient,
)


@dataclass(frozen=True)
class WSection:
    """A doubly symmetric I-shape (or channel) from the AISC Shapes Database.

    Fundamental properties are inputs; r_ts and c are derived. Frozen so it is
    hashable/immutable and safe to share across calculations.

    Args:
        plastic_section_modulus: Z_x (in^3)
        section_modulus_x_axis: S_x (in^3)
        second_moment_of_area_y_axis: I_y (in^4)
        warping_constant: C_w (in^6)
        radius_of_gyration_y_axis: r_y (in)
        torsional_constant: J (in^4)
        flange_centroid_distance: h_o (in)
        name: optional label, e.g. "W18X50"
        section_type: "i_shape" (c = 1.0) or "channel" (c per Eq. F2-8b)
    """

    plastic_section_modulus: float  # Z_x
    section_modulus_x_axis: float  # S_x
    second_moment_of_area_y_axis: float  # I_y
    warping_constant: float  # C_w
    radius_of_gyration_y_axis: float  # r_y
    torsional_constant: float  # J
    flange_centroid_distance: float  # h_o
    name: str = ""
    section_type: Literal["i_shape", "channel"] = "i_shape"

    @property
    def effective_radius_of_gyration(self) -> float:
        """r_ts, derived from I_y, C_w, S_x (Eq. F2-7)."""
        return calculate_effective_radius_of_gyration(
            self.second_moment_of_area_y_axis,
            self.warping_constant,
            self.section_modulus_x_axis,
        )

    @property
    def torsional_coefficient(self) -> float:
        """c: 1.0 for a doubly symmetric I-shape (Eq. F2-8a), else Eq. F2-8b."""
        if self.section_type == "i_shape":
            return 1.0
        return calculate_torsional_coefficient(
            self.flange_centroid_distance,
            self.second_moment_of_area_y_axis,
            self.warping_constant,
        )
