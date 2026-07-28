"""AISC shape section objects with a classmethod factory (kip-inch)."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, ClassVar

from .constants import YOUNGS_MODULUS_KSI


def _get(obj: Any, *names: str) -> float:
    """First attribute found on obj among names, as float.
    Args:
        obj: steelpy section object.
        names: candidate attribute spellings, tried in order.
    """
    for n in names:
        if hasattr(obj, n):
            return float(getattr(obj, n))
    raise AttributeError(f"none of {names!r} on {type(obj).__name__}")


def _normalize(designation: str) -> str:
    """steelpy attribute spelling: strip spaces, '.', '-', '/' -> '_'."""
    key = designation.strip().upper()
    for ch in (".", "-", "/", " "):
        key = key.replace(ch, "_")
    return key


def _fetch(collection: Any, key: str) -> Any:
    """Attribute-access lookup into a steelpy collection, or None."""
    return getattr(collection, key, None)


@dataclass(kw_only=True)
class Section:
    """Universal AISC properties shared by every rolled shape (kip-inch).
    Attributes:
        designation: AISC label, e.g. "W18X50".
        area: cross-sectional area A.
        second_moment_of_area_x_axis: I_x.
        second_moment_of_area_y_axis: I_y.
        section_modulus_x_axis: elastic S_x.
        plastic_section_modulus_x_axis: Z_x.
        plastic_section_modulus_y_axis: Z_y.
        radius_of_gyration_x_axis: r_x.
        radius_of_gyration_y_axis: r_y.
        torsional_constant: St-Venant J.
        raw: the underlying steelpy object, for any extra property.
    """

    designation: str
    area: float
    second_moment_of_area_x_axis: float
    second_moment_of_area_y_axis: float
    section_modulus_x_axis: float
    plastic_section_modulus_x_axis: float
    plastic_section_modulus_y_axis: float
    radius_of_gyration_x_axis: float
    radius_of_gyration_y_axis: float
    torsional_constant: float
    raw: Any = field(default=None, repr=False)
    aisc_flexure_section: ClassVar[str] = ""

    @property
    def name(self) -> str:
        """Alias for designation (e.g. for display / plotting labels)."""
        return self.designation

    @staticmethod
    def _base_kwargs(s: Any, designation: str) -> dict:
        """Universal properties common to every steelpy shape.
        Args:
            s: steelpy section object.
            designation: original AISC label as passed in.
        """
        return dict(
            designation=designation,
            area=_get(s, "area"),
            second_moment_of_area_x_axis=_get(s, "Ix"),
            second_moment_of_area_y_axis=_get(s, "Iy", "Ix"),  # round: Iy=Ix
            section_modulus_x_axis=_get(s, "Sx"),
            plastic_section_modulus_x_axis=_get(s, "Zx"),
            plastic_section_modulus_y_axis=_get(s, "Zy"),
            radius_of_gyration_x_axis=_get(s, "rx"),
            radius_of_gyration_y_axis=_get(s, "ry", "rx"),
            torsional_constant=_get(s, "J"),
            raw=s,
        )

    @classmethod
    def _build(cls, s: Any, designation: str) -> "Section":
        """Populate this specific subclass from a steelpy object.
        Overridden by each subclass to add its shape-specific fields.
        Args:
            s: steelpy section object.
            designation: original AISC label.
        """
        return cls(**cls._base_kwargs(s, designation))

    @classmethod
    def from_shape(cls, designation: str, aisc: Any = None) -> "Section":
        """Build the right Section subclass from a steelpy shape lookup.
        The shape family is inferred from the designation prefix (order
        matters: WT/MT/ST and HSS/PIPE are checked before the single-letter
        families). Round vs rectangular HSS is resolved by trying the
        rectangular table first, then the round table.
        Args:
            designation: AISC label such as "W18X50", "WT9X25", "L4X4X1/4",
                "HSS6X6X1/4", "HSS6.625X0.500", or "Pipe4STD".
            aisc: the steelpy `aisc` module; imported lazily if omitted.
        Returns:
            A populated Section subclass instance.
        """
        if aisc is None:
            from steelpy import aisc  # noqa: PLC0415
        key = _normalize(designation)
        up = key.upper()

        if up.startswith(("WT", "MT", "ST")):
            table = {"WT": aisc.WT_shapes, "MT": aisc.MT_shapes, "ST": aisc.ST_shapes}[up[:2]]
            s = _fetch(table, key)
            if s is None:
                raise KeyError(f"{designation!r} not found in {up[:2]}_shapes")
            return TSection._build(s, designation)

        if up.startswith("HSS"):
            s = _fetch(aisc.HSS_shapes, key)
            if s is not None:
                return HSSSection._build(s, designation)
            s = _fetch(aisc.HSS_R_shapes, key)
            if s is None:
                raise KeyError(f"{designation!r} not in HSS_shapes or HSS_R_shapes")
            return PipeSection._build(s, designation)

        if up.startswith("PIPE"):
            s = _fetch(aisc.PIPE_shapes, key) or _fetch(aisc.PIPE_shapes, designation.strip())
            if s is None:
                raise KeyError(f"{designation!r} not found in PIPE_shapes")
            return PipeSection._build(s, designation)

        if up.startswith("L"):
            s = _fetch(aisc.L_shapes, key)
            if s is None:
                raise KeyError(f"{designation!r} not found in L_shapes")
            return LSection._build(s, designation)

        fam = {"W": "W_shapes", "HP": "HP_shapes", "M": "M_shapes", "S": "S_shapes", "C": "C_shapes", "MC": "MC_shapes"}
        for prefix in ("HP", "MC", "W", "M", "S", "C"):  # longer prefixes first
            if up.startswith(prefix):
                s = _fetch(getattr(aisc, fam[prefix]), key)
                if s is None:
                    raise KeyError(f"{designation!r} not found in {fam[prefix]}")
                return WSection._build(s, designation)

        raise ValueError(f"unrecognized shape family for {designation!r}")


@dataclass(kw_only=True)
class WSection(Section):
    """Doubly symmetric I-shape (W, M, S, HP) or channel (C, MC) -> F2.
    Attributes:
        warping_constant: C_w.
        flange_centroid_distance: h_o.
    """

    warping_constant: float
    flange_centroid_distance: float
    aisc_flexure_section: ClassVar[str] = "F2"

    @classmethod
    def _build(cls, s: Any, designation: str) -> "WSection":
        return cls(
            **cls._base_kwargs(s, designation), warping_constant=_get(s, "Cw"), flange_centroid_distance=_get(s, "ho")
        )

    @property
    def effective_radius_of_gyration(self) -> float:
        """r_ts (Eq. F2-7), derived from I_y, C_w, S_x."""
        return sqrt(sqrt(self.second_moment_of_area_y_axis * self.warping_constant) / self.section_modulus_x_axis)

    @property
    def torsional_coefficient(self) -> float:
        """c = 1.0 for a doubly symmetric I-shape (Eq. F2-8a)."""
        return 1.0


@dataclass(kw_only=True)
class TSection(Section):
    """Tee (WT, MT, ST) -> F9.
    Attributes:
        depth: overall stem depth d.
        flange_width: b_f.
        flange_thickness: t_f.
        stem_thickness: t_w.
        warping_constant: C_w.
        centroid_to_extreme_stem_fiber: y_bar to stem tip.
    """

    depth: float
    flange_width: float
    flange_thickness: float
    stem_thickness: float
    warping_constant: float
    centroid_to_extreme_stem_fiber: float
    aisc_flexure_section: ClassVar[str] = "F9"

    @classmethod
    def _build(cls, s: Any, designation: str) -> "TSection":
        return cls(
            **cls._base_kwargs(s, designation),
            depth=_get(s, "d"),
            flange_width=_get(s, "bf"),
            flange_thickness=_get(s, "tf"),
            stem_thickness=_get(s, "tw"),
            warping_constant=_get(s, "Cw"),
            centroid_to_extreme_stem_fiber=_get(s, "y", "yp"),
        )


@dataclass(kw_only=True)
class LSection(Section):
    """Single angle -> F10 (principal-axis behavior).
    Attributes:
        leg_long: longer leg d.
        leg_short: shorter leg b.
        thickness: leg thickness t.
        minor_principal_radius_of_gyration: r_z.
        minor_principal_second_moment_of_area: I_z.
    """

    leg_long: float
    leg_short: float
    thickness: float
    minor_principal_radius_of_gyration: float
    minor_principal_second_moment_of_area: float
    aisc_flexure_section: ClassVar[str] = "F10"

    @classmethod
    def _build(cls, s: Any, designation: str) -> "LSection":
        return cls(
            **cls._base_kwargs(s, designation),
            leg_long=_get(s, "d"),
            leg_short=_get(s, "b"),
            thickness=_get(s, "T"),
            minor_principal_radius_of_gyration=_get(s, "rz"),
            minor_principal_second_moment_of_area=_get(s, "Iz"),
        )


@dataclass(kw_only=True)
class HSSSection(Section):
    """Rectangular / square HSS -> F7.
    Attributes:
        height: overall Ht.
        width: overall B.
        design_wall_thickness: t_des.
        hss_torsional_constant: HSS torsional constant C (not J).
    """

    height: float
    width: float
    design_wall_thickness: float
    hss_torsional_constant: float
    aisc_flexure_section: ClassVar[str] = "F7"

    @classmethod
    def _build(cls, s: Any, designation: str) -> "HSSSection":
        return cls(
            **cls._base_kwargs(s, designation),
            height=_get(s, "Ht", "H"),
            width=_get(s, "b"),
            design_wall_thickness=_get(s, "tdes", "tnom"),
            hss_torsional_constant=_get(s, "C", "J"),
        )

    @property
    def flange_slenderness(self) -> float:
        """Flat-width b/t (clear width ~ B - 3*t_des per AISC)."""
        return (self.width - 3.0 * self.design_wall_thickness) / self.design_wall_thickness

    @property
    def web_slenderness(self) -> float:
        """Flat-depth h/t (clear depth ~ Ht - 3*t_des per AISC)."""
        return (self.height - 3.0 * self.design_wall_thickness) / self.design_wall_thickness


@dataclass(kw_only=True)
class PipeSection(Section):
    """Round HSS or Pipe -> F8.
    Attributes:
        outside_diameter: OD.
        design_wall_thickness: t_des.
        hss_torsional_constant: torsional constant C.
    """

    outside_diameter: float
    design_wall_thickness: float
    hss_torsional_constant: float
    aisc_flexure_section: ClassVar[str] = "F8"

    @classmethod
    def _build(cls, s: Any, designation: str) -> "PipeSection":
        return cls(
            **cls._base_kwargs(s, designation),
            outside_diameter=_get(s, "OD", "D"),
            design_wall_thickness=_get(s, "tdes", "tnom"),
            hss_torsional_constant=_get(s, "C", "J"),
        )

    @property
    def diameter_thickness_ratio(self) -> float:
        """D/t governing F8 limits."""
        return self.outside_diameter / self.design_wall_thickness
