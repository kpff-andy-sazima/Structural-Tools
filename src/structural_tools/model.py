"""Modeling tools"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from Pynite import FEModel3D
from Pynite.LoadCombo import LoadCombo

from .general import DesignMethod
from .steel.constants import YOUNGS_MODULUS_KSI
from .steel.flexure import Bracing, evaluate_beam_flexure
from .typing import FloatLike

if TYPE_CHECKING:
    from .section import WSection


@dataclass
class ModelMaterial:
    name: str  # Material name
    E: float  # Modulus of elasticity
    nu: float  # Poisson's ratio
    rho: float  # Density
    fy: float  # Yield stress

    def __post_init__(self):
        self.G: float = self.E / (2 * (1 + self.nu))  # Shear modulus of elasticity


MATERIAL_STEEL_36_KSI_KIP_INCH: ModelMaterial = ModelMaterial(
    name="steel 36ksi",
    E=29000,
    nu=0.3,
    rho=489 / (12**3) / 1000,
    fy=36,
)

MATERIAL_STEEL_46_KSI_KIP_INCH: ModelMaterial = ModelMaterial(
    name="steel 46ksi",
    E=29000,
    nu=0.3,
    rho=489 / (12**3) / 1000,
    fy=46,
)

MATERIAL_STEEL_50_KSI_KIP_INCH: ModelMaterial = ModelMaterial(
    name="steel 50ksi",
    E=29000,
    nu=0.3,
    rho=489 / (12**3) / 1000,
    fy=50,
)


MATERIAL_WOOD_KIP_INCH: ModelMaterial = ModelMaterial(
    name="wood",
    E=1700,
    nu=0.3,
    rho=30 / (12**3) / 1000,
    fy=2.4,
)


class LoadCase(Enum):
    """ASCE 7 principal load cases. Value is the primary PyNite case label.

    SEISMIC spans two labels (E_v, E_h); the value is informational only since
    inclusion is driven by the enum member, not the label string.
    """

    DEAD = "D"
    LIVE = "L"
    LIVE_ROOF = "L_r"
    SNOW = "S"
    RAIN = "R"
    WIND = "W"
    SEISMIC = "E"


@dataclass(frozen=True)
class ComboSpec:
    """Declarative definition of one load combination.

    Attributes:
        name: Combo name passed to PyNite.
        factors: Mapping of load-case label to factor.
        tags: Combo tags.
        design_method: Restrict to a method; None means any method.
        service_flag: Key of the include_* gate, or None if not gated.
        always_include: If True, always add verbatim, bypassing all filtering
            (used for debug combos like dead-only and all-nominal).
    """

    name: str
    factors: dict[str, float]
    tags: tuple[str, ...]
    design_method: DesignMethod | None = None
    service_flag: str | None = None
    always_include: bool = False


LOAD_COMBINATIONS: tuple[ComboSpec, ...] = (
    ComboSpec("D", {"D": 1.0}, ("self weight", "debug"), always_include=True),
    ComboSpec(
        "all",
        {
            "D": 1.0,
            "L": 1.0,
            "L_r": 1.0,
            "S": 1.0,
            "W": 1.0,
            "R": 1.0,
            "E_v": 1.0,
            "E_h": 1.0,
        },
        ("all nominal", "debug"),
        always_include=True,
    ),
    # LRFD strength
    ComboSpec("1a", {"D": 1.4}, ("strength", "principal D"), DesignMethod.LRFD),
    ComboSpec("2a-1", {"D": 1.2, "L": 1.6, "L_r": 0.5}, ("strength", "principal L"), DesignMethod.LRFD),
    ComboSpec("2a-2", {"D": 1.2, "L": 1.6, "S": 0.3}, ("strength", "principal L"), DesignMethod.LRFD),
    ComboSpec("2a-3", {"D": 1.2, "L": 1.6, "R": 0.5}, ("strength", "principal L"), DesignMethod.LRFD),
    ComboSpec("3a-1", {"D": 1.2, "L_r": 1.6, "L": 1.0}, ("strength", "principal L_r"), DesignMethod.LRFD),
    ComboSpec("3a-2", {"D": 1.2, "S": 1.0, "L": 1.0}, ("strength", "principal S"), DesignMethod.LRFD),
    ComboSpec("3a-3", {"D": 1.2, "R": 1.6, "L": 1.0}, ("strength", "principal R"), DesignMethod.LRFD),
    ComboSpec("3a-4", {"D": 1.2, "L_r": 1.6, "W": 0.5}, ("strength", "principal L_r"), DesignMethod.LRFD),
    ComboSpec("3a-5", {"D": 1.2, "S": 1.0, "W": 0.5}, ("strength", "principal S"), DesignMethod.LRFD),
    ComboSpec("3a-6", {"D": 1.2, "R": 1.6, "W": 0.5}, ("strength", "principal R"), DesignMethod.LRFD),
    ComboSpec("4a-1", {"D": 1.2, "W": 1.0, "L": 1.0, "L_r": 0.5}, ("strength", "principal W"), DesignMethod.LRFD),
    ComboSpec("4a-2", {"D": 1.2, "W": 1.0, "L": 1.0, "S": 0.3}, ("strength", "principal W"), DesignMethod.LRFD),
    ComboSpec("4a-3", {"D": 1.2, "W": 1.0, "L": 1.0, "R": 0.5}, ("strength", "principal W"), DesignMethod.LRFD),
    ComboSpec("5a", {"D": 0.9, "W": -1.0}, ("strength", "principal W"), DesignMethod.LRFD),
    ComboSpec(
        "6", {"D": 1.2, "E_v": 1.0, "E_h": 1.0, "L": 1.0, "S": 0.15}, ("strength", "principal E"), DesignMethod.LRFD
    ),
    ComboSpec("7", {"D": 0.9, "E_v": -1.0, "E_h": 1.0}, ("strength", "principal E"), DesignMethod.LRFD),
    # ASD strength
    ComboSpec("1a", {"D": 1.0}, ("strength", "principal D"), DesignMethod.ASD),
    ComboSpec("2a", {"D": 1.0, "L": 1.0}, ("strength", "principal L"), DesignMethod.ASD),
    ComboSpec("3a-1", {"D": 1.0, "L_r": 1.0}, ("strength", "principal L_r"), DesignMethod.ASD),
    ComboSpec("3a-2", {"D": 1.0, "S": 0.7}, ("strength", "principal S"), DesignMethod.ASD),
    ComboSpec("3a-3", {"D": 1.0, "R": 1.0}, ("strength", "principal R"), DesignMethod.ASD),
    ComboSpec("4a-1", {"D": 1.0, "L": 0.75, "L_r": 0.75}, ("strength", "principal L"), DesignMethod.ASD),
    ComboSpec("4a-2", {"D": 1.0, "L": 0.75, "S": 0.525}, ("strength", "principal L"), DesignMethod.ASD),
    ComboSpec("4a-3", {"D": 1.0, "L": 0.75, "R": 0.75}, ("strength", "principal L"), DesignMethod.ASD),
    ComboSpec("5a", {"D": 1.0, "W": 0.6}, ("strength", "principal W"), DesignMethod.ASD),
    ComboSpec("6a-1", {"D": 1.0, "L": 0.75, "W": 0.45, "L_r": 0.75}, ("strength", "principal W"), DesignMethod.ASD),
    ComboSpec("6a-2", {"D": 1.0, "L": 0.75, "W": 0.45, "S": 0.525}, ("strength", "principal W"), DesignMethod.ASD),
    ComboSpec("6a-3", {"D": 1.0, "L": 0.75, "W": 0.45, "R": 0.75}, ("strength", "principal W"), DesignMethod.ASD),
    ComboSpec("7a", {"D": 0.6, "W": -0.6}, ("strength", "principal W"), DesignMethod.ASD),
    ComboSpec("8", {"D": 1.0, "E_v": 0.7, "E_h": 0.7}, ("strength", "principal E"), DesignMethod.ASD),
    ComboSpec(
        "9", {"D": 1.0, "E_v": 0.525, "E_h": 0.525, "L": 0.75, "S": 0.1}, ("strength", "principal E"), DesignMethod.ASD
    ),
    ComboSpec("10", {"D": 0.6, "E_v": -0.7, "E_h": 0.7}, ("strength", "principal E"), DesignMethod.ASD),
    # OSSC service / creep
    ComboSpec("L OSSC25", {"L": 1.0}, ("service", "OSSC25"), service_flag="ossc_service"),
    ComboSpec("L_r OSSC25", {"L_r": 1.0}, ("service", "OSSC25"), service_flag="ossc_service"),
    ComboSpec("S OSSC25", {"S": 1.0}, ("service", "OSSC25"), service_flag="ossc_service"),
    ComboSpec("W OSSC25", {"W": 1.0}, ("service", "OSSC25"), service_flag="ossc_service"),
    ComboSpec("0.5D+L OSSC25", {"D": 0.5, "L": 1.0}, ("creep", "OSSC25", "dry wood"), service_flag="ossc_creep"),
    ComboSpec("D+L OSSC25", {"D": 1.0, "L": 1.0}, ("creep", "OSSC25", "moist wood"), service_flag="ossc_creep"),
    # ASCE service / creep
    ComboSpec("CC.2-1a ASCE7-22", {"D": 1.0, "L": 1.0}, ("service", "ASCE7-22"), service_flag="asce_service"),
    ComboSpec("CC.2-1b ASCE7-22", {"D": 1.0, "S_ser": 1.0}, ("service", "ASCE7-22"), service_flag="asce_service"),
    ComboSpec("CC.2-2 ASCE7-22", {"D": 1.0, "L": 0.5}, ("creep", "ASCE7-22"), service_flag="asce_creep"),
)


# Map each PyNite load label to the LoadCase that governs it.
_LABEL_TO_CASE: dict[str, LoadCase] = {
    "D": LoadCase.DEAD,
    "L": LoadCase.LIVE,
    "L_r": LoadCase.LIVE_ROOF,
    "S": LoadCase.SNOW,
    "S_ser": LoadCase.SNOW,
    "R": LoadCase.RAIN,
    "W": LoadCase.WIND,
    "E_v": LoadCase.SEISMIC,
    "E_h": LoadCase.SEISMIC,
}


def create_base_model(
    design_method: DesignMethod = DesignMethod.LRFD,
    load_cases: LoadCase | list[LoadCase] | None = None,
    materials: ModelMaterial | list[ModelMaterial] = MATERIAL_STEEL_50_KSI_KIP_INCH,
    include_ossc_service_combos: bool = True,
    include_ossc_creep_combos: bool = True,
    include_asce_service_combos: bool = False,
    include_asce_creep_combos: bool = False,
) -> FEModel3D:
    """Build an FEModel3D preloaded with materials and load combinations.

    Combinations are added verbatim (never modified). A combo is added only when
    every load it references belongs to an active LoadCase, with dead always
    available. This guarantees no combo references an absent (zero) load while
    keeping the published factors exactly as defined. The dead-only and
    all-nominal debug combos are always added regardless of load_cases (they are
    tagged "debug" for easy filtering from graphs).

    Args:
        design_method: LRFD or ASD; selects the strength combo set.
        load_cases: Active load cases; a single member or list.
        materials: Material or list of materials to register.
        include_ossc_service_combos: Emit OSSC service combos (subset-filtered).
        include_ossc_creep_combos: Emit OSSC creep combos (subset-filtered).
        include_asce_service_combos: Emit ASCE service combos (subset-filtered).
        include_asce_creep_combos: Emit ASCE creep combos (subset-filtered).
    Returns:
        Configured FEModel3D instance.
    """
    base_model = FEModel3D()

    if isinstance(materials, ModelMaterial):
        materials = [materials]
    if isinstance(load_cases, LoadCase):
        load_cases = [load_cases]
    # Dead is always present in the model, so it is always an available load.
    available_cases = set(load_cases or []) | {LoadCase.DEAD}

    for material in materials:
        base_model.add_material(
            name=material.name,
            E=material.E,
            G=material.G,
            nu=material.nu,
            rho=material.rho,
            fy=material.fy,
        )

    flags = {
        "ossc_service": include_ossc_service_combos,
        "ossc_creep": include_ossc_creep_combos,
        "asce_service": include_asce_service_combos,
        "asce_creep": include_asce_creep_combos,
    }

    for spec in LOAD_COMBINATIONS:
        if spec.always_include:
            # Debug combos (dead-only, all-nominal) are always added verbatim.
            base_model.add_load_combo(spec.name, dict(spec.factors), list(spec.tags))
            continue
        if spec.design_method is not None and spec.design_method != design_method:
            continue
        if spec.service_flag is not None and not flags[spec.service_flag]:
            continue
        # Include only if every referenced load is available. Combo unchanged.
        if not all(_LABEL_TO_CASE[label] in available_cases for label in spec.factors):
            continue
        base_model.add_load_combo(spec.name, dict(spec.factors), list(spec.tags))

    return base_model


def _ensure_load_metadata(obj) -> None:
    """Ensure metadata storage exists."""

    if not hasattr(obj, "_load_metadata"):
        obj._load_metadata = defaultdict(list)


def add_named_node_load(
    model: FEModel3D,
    node_name: str,
    direction: str,
    p: float,
    case: str = "Case 1",
    name: str | None = None,
    category: str | None = None,
) -> None:
    """Add a named nodal load.

    Args:
        model: PyNite model.
        node_name: Node name.
        direction: Load direction.
        p: Load magnitude.
        case: Load case.
        name: Human-readable load name.
        category: Optional grouping category.
    """

    model.add_node_load(
        node_name=node_name,
        direction=direction,
        P=p,
        case=case,
    )

    node = model.nodes[node_name]

    _ensure_load_metadata(node)

    node._load_metadata["node_loads"].append({
        "direction": direction,
        "P": p,
        "case": case,
        "name": name or case,
        "category": category,
    })


def add_named_member_pt_load(
    model: FEModel3D,
    member_name: str,
    direction: str,
    p: float,
    x: float,
    case: str = "Case 1",
    name: str | None = None,
    category: str | None = None,
) -> None:
    """Add a named member point load.

    Args:
        model: PyNite model.
        member_name: Member name.
        direction: Load direction.
        p: Load magnitude.
        x: Location along member.
        case: Load case.
        name: Human-readable load name.
        category: Optional grouping category.
    """

    model.add_member_pt_load(
        member_name=member_name,
        direction=direction,
        P=p,
        x=x,
        case=case,
    )

    member = model.members[member_name]

    _ensure_load_metadata(member)

    member._load_metadata["point_loads"].append({
        "direction": direction,
        "P": p,
        "x": x,
        "case": case,
        "name": name or case,
        "category": category,
    })


def add_named_member_dist_load(
    model: FEModel3D,
    member_name: str,
    direction: str,
    w1: float,
    w2: float,
    x1: float | None = None,
    x2: float | None = None,
    case: str = "Case 1",
    name: str | None = None,
    category: str | None = None,
    self_weight: bool = False,
) -> None:
    """Add a named distributed member load.

    Args:
        model: PyNite model.
        member_name: Member name.
        direction: Load direction.
        w1: Start load magnitude.
        w2: End load magnitude.
        x1: Start location.
        x2: End location.
        case: Load case.
        name: Human-readable load name.
        category: Optional grouping category.
        self_weight: Indicates a self-weight load.
    """

    model.add_member_dist_load(
        member_name=member_name,
        direction=direction,
        w1=w1,
        w2=w2,
        x1=x1,
        x2=x2,
        case=case,
        self_weight=self_weight,
    )

    member = model.members[member_name]

    _ensure_load_metadata(member)

    member._load_metadata["dist_loads"].append({
        "direction": direction,
        "w1": w1,
        "w2": w2,
        "x1": x1,
        "x2": x2,
        "case": case,
        "name": name or case,
        "category": category,
        "self_weight": self_weight,
    })


def add_named_self_weight(
    model: FEModel3D,
    global_direction: str,
    factor: float,
    case: str = "D",
    name: str = "Self weight",
) -> None:
    """Add named self-weight loads."""

    for member in model.members.values():
        self_weight = factor * member.material.rho * member.section.A

        add_named_member_dist_load(
            model=model,
            member_name=member.name,
            direction=global_direction,
            w1=self_weight,
            w2=self_weight,
            case=case,
            name=name,
            self_weight=True,
        )


def get_load_combos_from_tags(model: FEModel3D, tags=str | list[str]) -> dict[str, LoadCombo]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        for name, combo in model.load_combos.items():
            if tag in combo.combo_tags:
                filtered_combos[name] = combo

    return filtered_combos


def get_moments_from_tags(
    model: FEModel3D,
    member_name: str = "m_1",
    tags: str | list[str] = "strength",
    direction: Literal["My", "Mz"] = "Mz",
    n_points: int = 500,
) -> tuple[list, list[list], list, dict]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        filtered_combos.update(get_load_combos_from_tags(model, tag))

    moments_dict = {}
    member = model.members[member_name]
    for combo in filtered_combos.values():
        x, M = member.moment_array(direction, n_points, combo_name=combo.name)
        combo_str = (
            combo.name + ": " + "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
        )
        combo_str = combo_str.replace("+ -", "-")
        moments_dict[combo.name] = {"moments_list": M, "label": combo_str}

    moments = [item["moments_list"] for item in moments_dict.values()]
    labels = [item["label"] for item in moments_dict.values()]

    return x, moments, labels, moments_dict


def get_deflections_from_tags(
    model: FEModel3D,
    member_name: str = "m_1",
    tags: str | list[str] = "service",
    direction: Literal["dx", "dy", "dz"] = "dy",
    n_points: int = 500,
) -> tuple[list, list[list], list, dict]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        filtered_combos.update(get_load_combos_from_tags(model, tag))

    deflections_dict = {}
    member = model.members[member_name]
    for combo in filtered_combos.values():
        x, M = member.deflection_array(direction, n_points, combo_name=combo.name)
        combo_str = (
            combo.name + ": " + "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
        )
        combo_str = combo_str.replace("+ -", "-")
        deflections_dict[combo.name] = {"deflections_list": M, "label": combo_str}

    deflections = [item["deflections_list"] for item in deflections_dict.values()]
    labels = [item["label"] for item in deflections_dict.values()]

    return x, deflections, labels, deflections_dict


def get_shears_from_tags(
    model: FEModel3D,
    member_name: str = "m_1",
    tags: str | list[str] = "strength",
    direction: Literal["Fy", "Fz"] = "Fy",
    n_points: int = 500,
) -> tuple[list, list[list], list, dict]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        filtered_combos.update(get_load_combos_from_tags(model, tag))

    shears_dict = {}
    member = model.members[member_name]
    for combo in filtered_combos.values():
        x, M = member.shear_array(direction, n_points, combo_name=combo.name)
        combo_str = (
            combo.name + ": " + "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
        )
        combo_str = combo_str.replace("+ -", "-")
        shears_dict[combo.name] = {"shears_list": M, "label": combo_str}

    shears = [item["shears_list"] for item in shears_dict.values()]
    labels = [item["label"] for item in shears_dict.values()]

    return x, shears, labels, shears_dict


def get_axial_from_tags(
    model: FEModel3D,
    member_name: str = "m_1",
    tags: str | list[str] = "strength",
    n_points: int = 500,
) -> tuple[list, list[list], list, dict]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        filtered_combos.update(get_load_combos_from_tags(model, tag))

    axial_dict = {}
    member = model.members[member_name]
    for combo in filtered_combos.values():
        x, M = member.shear_array(n_points, combo_name=combo.name)
        combo_str = (
            combo.name + ": " + "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
        )
        combo_str = combo_str.replace("+ -", "-")
        axial_dict[combo.name] = {"axial_list": M, "label": combo_str}

    axial = [item["axial_list"] for item in axial_dict.values()]
    labels = [item["label"] for item in axial_dict.values()]

    return x, axial, labels, axial_dict


def analyze_member_flexure(
    model: FEModel3D,
    section: WSection,
    member_name: str = "m_1",
    tags: str | list[str] = "strength",
    top_flange_brace_points: Sequence[FloatLike] = Bracing.CONTINUOUS,
    bottom_flange_brace_points: Sequence[FloatLike] = Bracing.UNBRACED,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
    positive_moment_compresses: str = "bottom",
    n_points: int = 500,
    moment_scale: float = 1 / 12,
    phi_b: float = 0.9,
    zero_atol=1e-6,
) -> pd.DataFrame:
    """Tidy one-row-per-segment DataFrame indexed by load combination.

    For each combo and each unbraced segment, record geometry, L_b, C_b,
    limit state, phi*M_n, M_u, and the demand/capacity ratio. Moments are
    scaled by `moment_scale` (default kip-in -> kip-ft).
    Numeric values within `zero_atol` of zero are snapped to exactly 0.0.
    Args:
        results_dict: mapping combo label -> list of FlexuralSegmentResult
            (exactly what your zip(labels, moments) loop builds).
        phi_b: flexural resistance factor applied to M_n (e.g. 0.90).
        moment_scale: multiplier applied to every moment column; default
            1/12 converts kip-in to kip-ft. Use 1.0 to keep kip-in.
        zero_atol: absolute tolerance below which numeric cells are set to 0.
    Returns:
        pandas.DataFrame indexed by "Load Combo", sorted by Load Combo then x_start.
    """
    x, moments, labels, moments_dict = get_moments_from_tags(
        model=model, member_name=member_name, tags=tags, n_points=n_points
    )

    results_dict = {}
    for combo_name, moment_list in zip(labels, moments):
        results_dict[combo_name] = evaluate_beam_flexure(
            section=section,
            yield_stress=model.members[member_name].material.fy,
            positions=x,
            moments=moment_list,
            top_flange_brace_points=top_flange_brace_points,
            bottom_flange_brace_points=bottom_flange_brace_points,
            youngs_modulus=youngs_modulus,
            positive_moment_compresses=positive_moment_compresses,
        )

    rows = []
    for combo, results in results_dict.items():
        for r in results:
            ltb = r.lateral_torsional_buckling
            phi_Mn = phi_b * ltb.nominal_moment
            rows.append({
                "Load Combo": combo,
                "flange": r.flange,
                "segment start": r.x_start,
                "segment end": r.x_end,
                "unbraced length": ltb.unbraced_length,
                "ltb modification factor": r.ltb_modification_factor,
                "limit region": ltb.region,
                "moment demand": r.moment_demand * moment_scale,
                "factored moment capacity": phi_Mn * moment_scale,
                "DCR": r.moment_demand / phi_Mn,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["Load Combo", "segment start"])

    # snap near-zero floats to exactly 0.0 (avoids -0.0 and 1e-16 noise)
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask(np.isclose(df[num], 0.0, atol=zero_atol), 0.0)

    return df.set_index("Load Combo")
