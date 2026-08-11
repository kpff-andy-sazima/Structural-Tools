"""Section property bag and derived quantities for AISC 360-22 Chapter F.

Chapter F needs far more than a WSection carries (S_xc/S_xt, h_c, b_fc, r_t,
beta_w, ...), and several of those are derived rather than tabulated. This
module holds one flat, optional-everything container plus the derivations, so
each Chapter F section can ask for exactly what it needs and skip a limit state
cleanly when a property is absent.

Units: kip, inch.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from math import pi, sqrt
from typing import TYPE_CHECKING, Optional

from ...typing import FloatLike
from .common import ShapeGroup

if TYPE_CHECKING:
    from ..section import WSection


def _clamp(value: float, low: float, high: float) -> float:
    """Bound `value` to [low, high]."""
    return min(high, max(low, value))


@dataclass
class SectionProperties:
    """Flat section-property bag consumed by every Chapter F section.

    All values are optional: a limit state that needs a missing property is
    reported as not-evaluated rather than raising. Call `derived()` to fill in
    the quantities Chapter F computes from the tabulated ones.

    Units: in, in^2, in^3, in^4, in^6.
    """

    group: ShapeGroup
    name: str = ""

    # --- gross geometry ---------------------------------------------------
    area: Optional[float] = None  # A_g
    depth: Optional[float] = None  # d
    flange_width: Optional[float] = None  # b_f
    flange_thickness: Optional[float] = None  # t_f
    web_thickness: Optional[float] = None  # t_w
    web_depth: Optional[float] = None  # h (B4.1b clear depth)
    web_depth_compression: Optional[float] = None  # h_c
    web_depth_plastic: Optional[float] = None  # h_p
    flange_centroid_distance: Optional[float] = None  # h_o
    fillet_distance: Optional[float] = None  # k_des
    outside_diameter: Optional[float] = None  # D (round HSS)
    box_height: Optional[float] = None  # H (rect HSS/box)
    box_width: Optional[float] = None  # B
    design_wall_thickness: Optional[float] = None  # t_des
    leg_width: Optional[float] = None  # b, angle leg in compression
    leg_thickness: Optional[float] = None  # t, angle leg
    bar_depth: Optional[float] = None  # d, bar in the plane of bending
    bar_width: Optional[float] = None  # t, bar out of plane

    # --- width-to-thickness ratios (Table B4.1b) --------------------------
    flange_slenderness: Optional[float] = None  # b_f/2t_f, b/t, b/t_f
    web_slenderness: Optional[float] = None  # h/t_w, h/t, d/t_w
    diameter_slenderness: Optional[float] = None  # D/t

    # --- section moduli and inertias --------------------------------------
    second_moment_x: Optional[float] = None  # I_x
    second_moment_y: Optional[float] = None  # I_y
    second_moment_compression_flange: Optional[float] = None  # I_yc
    plastic_modulus_x: Optional[float] = None  # Z_x
    plastic_modulus_y: Optional[float] = None  # Z_y
    section_modulus_x: Optional[float] = None  # S_x
    section_modulus_y: Optional[float] = None  # S_y
    section_modulus_compression: Optional[float] = None  # S_xc
    section_modulus_tension: Optional[float] = None  # S_xt
    section_modulus_compression_toe: Optional[float] = None  # S_c (angles)
    section_modulus_principal: Optional[float] = None  # S_w (angles)
    radius_of_gyration_x: Optional[float] = None  # r_x
    radius_of_gyration_y: Optional[float] = None  # r_y
    radius_of_gyration_z: Optional[float] = None  # r_z (angles)
    effective_radius_of_gyration: Optional[float] = None  # r_ts, Eq. F2-7
    compression_flange_radius: Optional[float] = None  # r_t, Eq. F4-11
    torsional_constant: Optional[float] = None  # J
    warping_constant: Optional[float] = None  # C_w
    torsional_coefficient: Optional[float] = None  # c, Eq. F2-8
    centroid_distance: Optional[float] = None  # y-bar (tees)
    angle_asymmetry: Optional[float] = None  # beta_w, Eq. F10-4

    # --- compression-flange geometry (built-up / singly symmetric) --------
    compression_flange_width: Optional[float] = None  # b_fc
    compression_flange_thickness: Optional[float] = None  # t_fc
    web_area_ratio: Optional[float] = None  # a_w, Eq. F4-12
    flange_buckling_coefficient: Optional[float] = None  # k_c
    limiting_flange_stress: Optional[float] = None  # F_L, Eq. F4-6

    # --- flags and user-supplied stresses ---------------------------------
    singly_symmetric: bool = False
    welded: bool = False  # built-up: Table B4.1b uses the k_c form for lambda_r
    box_section: bool = False  # True -> Eq. F7-5 instead of F7-4
    equal_leg_angle: bool = True
    critical_stress_ltb: Optional[float] = None  # F_cr for F12-3
    critical_stress_local: Optional[float] = None  # F_cr for F12-4

    # --- F13.1 tension-flange rupture -------------------------------------
    gross_flange_area: Optional[float] = None  # A_fg
    net_flange_area: Optional[float] = None  # A_fn

    @classmethod
    def from_wsection(cls, section: WSection, **overrides) -> SectionProperties:
        """Build from an existing WSection, mapping its long attribute names.

        Args:
            section: WSection (or anything exposing the same attribute names).
            **overrides: any SectionProperties field to set or replace.
        """
        mapping = {
            "area": "area",
            "depth": "depth",
            "flange_width": "flange_width",
            "flange_thickness": "flange_thickness",
            "web_thickness": "web_thickness",
            "flange_centroid_distance": "flange_centroid_distance",
            "second_moment_x": "second_moment_of_area_x_axis",
            "second_moment_y": "second_moment_of_area_y_axis",
            "plastic_modulus_x": "plastic_section_modulus_x_axis",
            "plastic_modulus_y": "plastic_section_modulus_y_axis",
            "section_modulus_x": "section_modulus_x_axis",
            "section_modulus_y": "section_modulus_y_axis",
            "radius_of_gyration_x": "radius_of_gyration_x_axis",
            "radius_of_gyration_y": "radius_of_gyration_y_axis",
            "torsional_constant": "torsional_constant",
            "warping_constant": "warping_constant",
            "effective_radius_of_gyration": "effective_radius_of_gyration",
            "torsional_coefficient": "torsional_coefficient",
        }
        values = {}
        for own, other in mapping.items():
            value = getattr(section, other, None)
            if value is not None:
                values[own] = float(value)
        values.setdefault("name", getattr(section, "designation", "") or "")
        values.setdefault("group", ShapeGroup.I_SHAPE)
        values.update(overrides)
        return cls(**values)

    def derived(self, yield_stress: FloatLike, youngs_modulus: FloatLike) -> SectionProperties:
        """Return a copy with every Chapter F derived property filled in.

        Fills h, h_c, h_p, h_o, r_ts (F2-7), c (F2-8), S_xc/S_xt, I_yc, b_fc,
        t_fc, a_w (F4-12), r_t (F4-11), F_L (F4-6), k_c and bar section moduli.
        Values already present are never overwritten.

        Args:
            yield_stress: F_y (ksi)
            youngs_modulus: E (ksi)
        """
        F_y = float(yield_stress)
        properties = replace(self)
        _derive_web_geometry(properties)
        _derive_torsion(properties)
        _derive_section_moduli(properties)
        _derive_compression_flange(properties, F_y)
        _derive_bars(properties)
        return properties

    def get(self, name: str) -> Optional[float]:
        """Attribute access by field name, returning None when absent."""
        return getattr(self, name, None)

    def present(self, *names: str) -> bool:
        """True when every named field holds a finite value."""
        return all(getattr(self, name, None) is not None for name in names)

    def as_dict(self) -> dict:
        """Field-name to value mapping, dropping Nones."""
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}


# --------------------------------------------------------------------------- #
# Derivations                                                                  #
# --------------------------------------------------------------------------- #
def _derive_web_geometry(p: SectionProperties) -> None:
    """Fill h, h_c, h_p and h_o."""
    if p.web_depth is None:
        if p.web_slenderness is not None and p.web_thickness is not None:
            p.web_depth = p.web_slenderness * p.web_thickness
        elif p.depth is not None and p.fillet_distance is not None:
            p.web_depth = p.depth - 2 * p.fillet_distance
        elif p.flange_centroid_distance is not None and p.flange_thickness is not None:
            p.web_depth = p.flange_centroid_distance - p.flange_thickness
    if p.web_depth_compression is None:
        p.web_depth_compression = p.web_depth  # h_c = h for doubly symmetric
    if p.web_depth_plastic is None:
        p.web_depth_plastic = p.web_depth_compression
    if p.flange_centroid_distance is None and p.depth is not None and p.flange_thickness is not None:
        p.flange_centroid_distance = p.depth - p.flange_thickness
    if p.web_slenderness is None and p.web_depth is not None and p.web_thickness:
        p.web_slenderness = p.web_depth / p.web_thickness


def _derive_torsion(p: SectionProperties) -> None:
    """Fill r_ts (Eq. F2-7) and the coefficient c (Eq. F2-8a / F2-8b)."""
    if (
        p.effective_radius_of_gyration is None
        and p.second_moment_y is not None
        and p.warping_constant is not None
        and p.section_modulus_x
    ):
        p.effective_radius_of_gyration = sqrt(
            sqrt(p.second_moment_y * p.warping_constant) / p.section_modulus_x
        )
    if p.torsional_coefficient is None:
        if (
            p.group is ShapeGroup.CHANNEL
            and p.flange_centroid_distance is not None
            and p.second_moment_y is not None
            and p.warping_constant
        ):
            # Eq. F2-8b
            p.torsional_coefficient = (
                p.flange_centroid_distance / 2 * sqrt(p.second_moment_y / p.warping_constant)
            )
        else:
            p.torsional_coefficient = 1.0  # Eq. F2-8a


def _derive_section_moduli(p: SectionProperties) -> None:
    """Fill S_xc / S_xt (splitting I_x at the centroid for tees) and I_yc."""
    if p.group is ShapeGroup.TEE and p.second_moment_x is not None and p.centroid_distance:
        if p.section_modulus_compression is None:
            p.section_modulus_compression = p.second_moment_x / p.centroid_distance
        if p.section_modulus_tension is None and p.depth is not None:
            p.section_modulus_tension = p.second_moment_x / (p.depth - p.centroid_distance)
        if p.section_modulus_x is None:
            p.section_modulus_x = p.section_modulus_tension  # AISC tabulates S_x at the stem tip
    if p.section_modulus_compression is None:
        p.section_modulus_compression = p.section_modulus_x
    if p.section_modulus_tension is None:
        p.section_modulus_tension = p.section_modulus_x
    if p.second_moment_compression_flange is None and p.second_moment_y is not None:
        is_i = p.group in (ShapeGroup.I_SHAPE, ShapeGroup.BUILT_UP_I)
        p.second_moment_compression_flange = (
            p.second_moment_y / 2 if is_i and not p.singly_symmetric else p.second_moment_y
        )


def _derive_compression_flange(p: SectionProperties, yield_stress: float) -> None:
    """Fill b_fc, t_fc, a_w (F4-12), r_t (F4-11), F_L (F4-6) and k_c."""
    if p.compression_flange_width is None:
        p.compression_flange_width = p.flange_width
    if p.compression_flange_thickness is None:
        p.compression_flange_thickness = p.flange_thickness
    if (
        p.web_area_ratio is None
        and p.web_depth_compression is not None
        and p.web_thickness is not None
        and p.compression_flange_width
        and p.compression_flange_thickness
    ):
        # Eq. F4-12, a_w shall not exceed 10
        p.web_area_ratio = _clamp(
            (p.web_depth_compression * p.web_thickness)
            / (p.compression_flange_width * p.compression_flange_thickness),
            0.0,
            10.0,
        )
    if (
        p.compression_flange_radius is None
        and p.compression_flange_width is not None
        and p.web_area_ratio is not None
    ):
        # Eq. F4-11
        p.compression_flange_radius = p.compression_flange_width / sqrt(
            12 * (1 + p.web_area_ratio / 6)
        )
    if (
        p.limiting_flange_stress is None
        and p.section_modulus_tension is not None
        and p.section_modulus_compression
    ):
        ratio = p.section_modulus_tension / p.section_modulus_compression
        # Eq. F4-6a / F4-6b
        p.limiting_flange_stress = (
            0.7 * yield_stress if ratio >= 0.7 else max(yield_stress * ratio, 0.5 * yield_stress)
        )
    if p.flange_buckling_coefficient is None and p.web_slenderness:
        # k_c, footnote to Table B4.1b: 0.35 <= k_c <= 0.76
        p.flange_buckling_coefficient = _clamp(4 / sqrt(p.web_slenderness), 0.35, 0.76)


def _derive_bars(p: SectionProperties) -> None:
    """Fill Z and S for rectangular bars and rounds from their dimensions."""
    if p.group is ShapeGroup.ROUND_BAR and p.bar_depth:
        d = p.bar_depth
        if p.plastic_modulus_x is None:
            p.plastic_modulus_x = p.plastic_modulus_y = d**3 / 6
        if p.section_modulus_x is None:
            p.section_modulus_x = p.section_modulus_y = pi * d**3 / 32
        if p.area is None:
            p.area = pi * d**2 / 4
    elif p.group is ShapeGroup.RECT_BAR and p.bar_depth and p.bar_width:
        d, t = p.bar_depth, p.bar_width
        if p.plastic_modulus_x is None:
            p.plastic_modulus_x = t * d**2 / 4
            p.plastic_modulus_y = d * t**2 / 4
        if p.section_modulus_x is None:
            p.section_modulus_x = t * d**2 / 6
            p.section_modulus_y = d * t**2 / 6
        if p.area is None:
            p.area = d * t
    if p.section_modulus_compression is None:
        p.section_modulus_compression = p.section_modulus_x
    if p.section_modulus_tension is None:
        p.section_modulus_tension = p.section_modulus_x
