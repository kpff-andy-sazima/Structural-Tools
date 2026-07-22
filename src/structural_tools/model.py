"""Modeling tools"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

from Pynite import FEModel3D

from .asce import DesignMethod

from enum import Enum, auto


class PrincipalLoad(Enum):
    DEAD = auto()
    LIVE = auto()
    LIVE_ROOF = auto()
    SNOW = auto()
    RAIN = auto()
    WIND = auto()
    SEISMIC = auto()


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


def create_base_model(
    design_method: DesignMethod = DesignMethod.LRFD,
    principal_loads: PrincipalLoad | list[PrincipalLoad] | None = None,
    materials: ModelMaterial | list[ModelMaterial] = MATERIAL_STEEL_50_KSI_KIP_INCH,
    include_ossc_service_combos: bool = True,
    include_ossc_creep_combos: bool = True,
    include_asce_service_combos: bool = False,
    include_asce_creep_combos: bool = False,
) -> FEModel3D:
    base_model = FEModel3D()

    if isinstance(materials, ModelMaterial):
        materials = [materials]

    if isinstance(principal_loads, PrincipalLoad):
        principal_loads = [principal_loads]

    for material in materials:
        base_model.add_material(
            name=material.name, E=material.E, G=material.G, nu=material.nu, rho=material.rho, fy=material.fy
        )

    # Load combinations
    load_combinations = []

    # Baseline load combinations
    load_combinations += [
        ("D", {"D": 1.0}, ["self weight", "debug"]),
        (
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
            ["all nominal", "debug"],
        ),
    ]

    # LRFD combos
    if design_method == DesignMethod.LRFD:
        if PrincipalLoad.DEAD in principal_loads:
            load_combinations += [("1a", {"D": 1.4}, ["strength", "principal D"])]
        if PrincipalLoad.LIVE in principal_loads:
            load_combinations += [
                ("2a-1", {"D": 1.2, "L": 1.6, "L_r": 0.5}, ["strength", "principal L"]),
                ("2a-2", {"D": 1.2, "L": 1.6, "S": 0.3}, ["strength", "principal L"]),
                ("2a-3", {"D": 1.2, "L": 1.6, "R": 0.5}, ["strength", "principal L"]),
            ]
        if PrincipalLoad.LIVE_ROOF in principal_loads:
            load_combinations += [
                ("3a-1", {"D": 1.2, "L_r": 1.6, "L": 1.0}, ["strength", "principal L_r"]),
                ("3a-4", {"D": 1.2, "L_r": 1.6, "W": 0.5}, ["strength", "principal L_r"]),
            ]
        if PrincipalLoad.SNOW in principal_loads:
            load_combinations += [
                ("3a-2", {"D": 1.2, "S": 1.0, "L": 1.0}, ["strength", "principal S"]),
                ("3a-5", {"D": 1.2, "S": 1.0, "W": 0.5}, ["strength", "principal S"]),
            ]
        if PrincipalLoad.RAIN in principal_loads:
            load_combinations += [
                ("3a-3", {"D": 1.2, "R": 1.6, "L": 1.0}, ["strength", "principal R"]),
                ("3a-6", {"D": 1.2, "R": 1.6, "W": 0.5}, ["strength", "principal R"]),
            ]
        if PrincipalLoad.WIND in principal_loads:
            load_combinations += [
                ("4a-1", {"D": 1.2, "W": 1.0, "L": 1.0, "L_r": 0.5}, ["strength", "principal W"]),
                ("4a-2", {"D": 1.2, "W": 1.0, "L": 1.0, "S": 0.3}, ["strength", "principal W"]),
                ("4a-3", {"D": 1.2, "W": 1.0, "L": 1.0, "R": 0.5}, ["strength", "principal W"]),
                ("5a", {"D": 0.9, "W": -1.0}, ["strength", "principal W"]),
            ]
        if PrincipalLoad.SEISMIC in principal_loads:
            load_combinations += [
                ("6", {"D": 1.2, "E_v": 1.0, "E_h": 1.0, "L": 1.0, "S": 0.15}, ["strength", "principal E"]),
                ("7", {"D": 0.9, "E_v": -1.0, "E_h": 1.0}, ["strength", "principal E"]),
            ]
    elif design_method == DesignMethod.ASD:
        if PrincipalLoad.DEAD in principal_loads:
            load_combinations += [("1a", {"D": 1.0}, ["strength", "principal D"])]
        if PrincipalLoad.LIVE in principal_loads:
            load_combinations += [
                ("2a", {"D": 1.0, "L": 1.0}, ["strength", "principal L"]),
                ("4a-1", {"D": 1.0, "L": 0.75, "L_r": 0.75}, ["strength", "principal L"]),
                ("4a-2", {"D": 1.0, "L": 0.75, "S": 0.525}, ["strength", "principal L"]),
                ("4a-3", {"D": 1.0, "L": 0.75, "R": 0.75}, ["strength", "principal L"]),
            ]
        if PrincipalLoad.LIVE_ROOF in principal_loads:
            load_combinations += [
                ("3a-1", {"D": 1.0, "L_r": 1.0}, ["strength", "principal L_r"]),
            ]
        if PrincipalLoad.SNOW in principal_loads:
            load_combinations += [
                ("3a-2", {"D": 1.0, "S": 0.7}, ["strength", "principal S"]),
            ]
        if PrincipalLoad.RAIN in principal_loads:
            load_combinations += [
                ("3a-3", {"D": 1.0, "R": 1.0}, ["strength", "principal R"]),
            ]
        if PrincipalLoad.WIND in principal_loads:
            load_combinations += [
                ("5a", {"D": 1.0, "W": 0.6}, ["strength", "principal W"]),
                ("6a-1", {"D": 1.0, "L": 0.75, "W": 0.45, "L_r": 0.75}, ["strength", "principal W"]),
                ("6a-2", {"D": 1.0, "L": 0.75, "W": 0.45, "S": 0.525}, ["strength", "principal W"]),
                ("6a-3", {"D": 1.0, "L": 0.75, "W": 0.45, "R": 0.75}, ["strength", "principal W"]),
                ("7a", {"D": 0.6, "W": -0.6}, ["strength", "principal W"]),
            ]
        if PrincipalLoad.SEISMIC in principal_loads:
            load_combinations += [
                ("8", {"D": 1.0, "E_v": 0.7, "E_h": 0.7}, ["strength", "principal E"]),
                ("9", {"D": 1.0, "E_v": 0.525, "E_h": 0.525, "L": 0.75, "S": 0.1}, ["strength", "principal E"]),
                ("10", {"D": 0.6, "E_v": -0.7, "E_h": 0.7}, ["strength", "principal E"]),
            ]
    else:
        raise ValueError("Design Method should be LRFD or ASD")

    # Add service load combos
    if include_ossc_service_combos:
        load_combinations += [
            ("L OSSC25", {"L": 1.0}, ["service OSSC25"]),
            ("L_r OSSC25", {"L_r": 1.0}, ["service OSSC25"]),
            ("S OSSC25", {"S": 1.0}, ["service OSSC25"]),
            ("W OSSC25", {"W": 1.0}, ["service OSSC25"]),
        ]
    if include_ossc_creep_combos:
        load_combinations += [
            ("0.5D+L OSSC25", {"D": 0.5, "L": 1.0}, ["creep OSSC25 dry wood"]),
            ("D+L OSSC25", {"D": 1.0, "L": 1.0}, ["creep OSSC25 moist wood"]),
        ]
    if include_asce_service_combos:
        load_combinations += [
            ("CC.2-1a ASCE7-22", {"D": 1.0, "L": 1.0}, ["service ASCE7-22"]),
            ("CC.2-1b ASCE7-22", {"D": 1.0, "S_ser": 1.0}, ["service ASCE7-22"]),
        ]
    if include_asce_creep_combos:
        load_combinations += [("CC.2-2 ASCE7-22", {"D": 1.0, "L": 0.5}, ["creep ASCE7-22"])]

    for name, factors, tags in load_combinations:
        base_model.add_load_combo(name, factors, tags)

    return deepcopy(base_model)


def _ensure_load_metadata(obj) -> None:
    """Ensure metadata storage exists."""

    if not hasattr(obj, "_load_metadata"):
        obj._load_metadata = defaultdict(list)


def add_named_node_load(
    model,
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
    model,
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
    model,
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
    model,
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
