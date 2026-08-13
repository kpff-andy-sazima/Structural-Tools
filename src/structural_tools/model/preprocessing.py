"""Modeling tools"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from Pynite import FEModel3D

from .. import DesignMethod
from ..asce import LoadCase
from ..steel.section import Section


@dataclass
class ModelMaterial:
    name: str  # Material name
    E: float  # Modulus of elasticity
    nu: float  # Poisson's ratio
    rho: float  # Density
    fy: float  # Yield stress

    def __post_init__(self):
        self.G: float = self.E / (2 * (1 + self.nu))  # Shear modulus of elasticity


MATERIAL_RIGID: ModelMaterial = ModelMaterial(
    name="mat_rigid",
    E=1e9,
    nu=0.0,
    rho=1e-18,
    fy=1e9,
)

MATERIAL_STEEL_36_KSI_KIP_INCH: ModelMaterial = ModelMaterial(
    name="mat_steel_36ksi",
    E=29000,
    nu=0.3,
    rho=490 / (12**3) / 1000,
    fy=36,
)

MATERIAL_STEEL_46_KSI_KIP_INCH: ModelMaterial = ModelMaterial(
    name="mat_steel_46ksi",
    E=29000,
    nu=0.3,
    rho=490 / (12**3) / 1000,
    fy=46,
)

MATERIAL_STEEL_50_KSI_KIP_INCH: ModelMaterial = ModelMaterial(
    name="mat_steel_50ksi",
    E=29000,
    nu=0.3,
    rho=490 / (12**3) / 1000,
    fy=50,
)

MATERIAL_STEEL_36_KSI_0_8_E_KIP_INCH: ModelMaterial = ModelMaterial(
    name="mat_steel_36ksi",
    E=29000 * 0.8,
    nu=0.3,
    rho=490 / (12**3) / 1000,
    fy=36,
)

MATERIAL_STEEL_46_KSI_0_8_E_KIP_INCH: ModelMaterial = ModelMaterial(
    name="mat_steel_46ksi",
    E=29000 * 0.8,
    nu=0.3,
    rho=490 / (12**3) / 1000,
    fy=46,
)

MATERIAL_STEEL_50_KSI_0_8_E_KIP_INCH: ModelMaterial = ModelMaterial(
    name="mat_steel_50ksi",
    E=29000 * 0.8,
    nu=0.3,
    rho=490 / (12**3) / 1000,
    fy=50,
)

MATERIAL_WOOD_KIP_INCH: ModelMaterial = ModelMaterial(
    name="wood",
    E=1700,
    nu=0.3,
    rho=30 / (12**3) / 1000,
    fy=2.4,
)


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
            # "E_h": 1.0,
            "E_hx": 1.0,
            "E_hy": 1.0,
        },
        ("all nominal", "debug"),
        always_include=True,
    ),
    # LRFD strength
    ComboSpec("1a", {"D": 1.4}, ("strength", "principal D"), DesignMethod.LRFD),
    ComboSpec("2a-1", {"D": 1.2, "L": 1.6}, ("strength", "principal L"), DesignMethod.LRFD),
    ComboSpec("2a-2", {"D": 1.2, "L": 1.6, "L_r": 0.5}, ("strength", "principal L"), DesignMethod.LRFD),
    ComboSpec("2a-3", {"D": 1.2, "L": 1.6, "S": 0.3}, ("strength", "principal L"), DesignMethod.LRFD),
    ComboSpec("2a-4", {"D": 1.2, "L": 1.6, "R": 0.5}, ("strength", "principal L"), DesignMethod.LRFD),
    ComboSpec("3a-1", {"D": 1.2, "L_r": 1.6, "L": 1.0}, ("strength", "principal L_r"), DesignMethod.LRFD),
    ComboSpec("3a-2", {"D": 1.2, "S": 1.0, "L": 1.0}, ("strength", "principal S"), DesignMethod.LRFD),
    ComboSpec("3a-3", {"D": 1.2, "R": 1.6, "L": 1.0}, ("strength", "principal R"), DesignMethod.LRFD),
    ComboSpec("3a-4", {"D": 1.2, "L_r": 1.6, "W": 0.5}, ("strength", "principal L_r"), DesignMethod.LRFD),
    ComboSpec("3a-5", {"D": 1.2, "S": 1.0, "W": 0.5}, ("strength", "principal S"), DesignMethod.LRFD),
    ComboSpec("3a-6", {"D": 1.2, "R": 1.6, "W": 0.5}, ("strength", "principal R"), DesignMethod.LRFD),
    ComboSpec("4a-1", {"D": 1.2, "W": 1.0, "L": 1.0, "L_r": 0.5}, ("strength", "principal W"), DesignMethod.LRFD),
    ComboSpec("4a-2", {"D": 1.2, "W": 1.0, "L": 1.0, "S": 0.3}, ("strength", "principal W"), DesignMethod.LRFD),
    ComboSpec("4a-3", {"D": 1.2, "W": 1.0, "L": 1.0, "R": 0.5}, ("strength", "principal W"), DesignMethod.LRFD),
    ComboSpec("5a", {"D": 0.9, "W": -1.0}, ("strength", "principal W", "uplift"), DesignMethod.LRFD),
    # ComboSpec("6-1", {"D": 1.2, "E_v": 1.0, "E_h": 1.0, "L": 1.0}, ("strength", "principal E"), DesignMethod.LRFD),
    ComboSpec("6-1x", {"D": 1.2, "E_v": 1.0, "E_hx": 1.0, "L": 1.0}, ("strength", "principal E"), DesignMethod.LRFD),
    ComboSpec("6-1y", {"D": 1.2, "E_v": 1.0, "E_hy": 1.0, "L": 1.0}, ("strength", "principal E"), DesignMethod.LRFD),
    # ComboSpec(
    #     "6-2", {"D": 1.2, "E_v": 1.0, "E_h": 1.0, "L": 1.0, "S": 0.15}, ("strength", "principal E"), DesignMethod.LRFD
    # ),
    ComboSpec(
        "6-2x", {"D": 1.2, "E_v": 1.0, "E_hx": 1.0, "L": 1.0, "S": 0.15}, ("strength", "principal E"), DesignMethod.LRFD
    ),
    ComboSpec(
        "6-2y", {"D": 1.2, "E_v": 1.0, "E_hy": 1.0, "L": 1.0, "S": 0.15}, ("strength", "principal E"), DesignMethod.LRFD
    ),
    # ComboSpec("7", {"D": 0.9, "E_v": -1.0, "E_h": 1.0}, ("strength", "principal E", "uplift"), DesignMethod.LRFD),
    ComboSpec("7x", {"D": 0.9, "E_v": -1.0, "E_hx": 1.0}, ("strength", "principal E", "uplift"), DesignMethod.LRFD),
    ComboSpec("7y", {"D": 0.9, "E_v": -1.0, "E_hy": 1.0}, ("strength", "principal E", "uplift"), DesignMethod.LRFD),
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
    ComboSpec("7a", {"D": 0.6, "W": -0.6}, ("strength", "principal W", "uplift"), DesignMethod.ASD),
    # ComboSpec("8", {"D": 1.0, "E_v": 0.7, "E_h": 0.7}, ("strength", "principal E"), DesignMethod.ASD),
    ComboSpec("8x", {"D": 1.0, "E_v": 0.7, "E_hx": 0.7}, ("strength", "principal E"), DesignMethod.ASD),
    ComboSpec("8y", {"D": 1.0, "E_v": 0.7, "E_hy": 0.7}, ("strength", "principal E"), DesignMethod.ASD),
    # ComboSpec("9-1", {"D": 1.0, "E_v": 0.525, "E_h": 0.525, "L": 0.75}, ("strength", "principal E"), DesignMethod.ASD),
    ComboSpec(
        "9-1x", {"D": 1.0, "E_v": 0.525, "E_hx": 0.525, "L": 0.75}, ("strength", "principal E"), DesignMethod.ASD
    ),
    ComboSpec(
        "9-1y", {"D": 1.0, "E_v": 0.525, "E_hy": 0.525, "L": 0.75}, ("strength", "principal E"), DesignMethod.ASD
    ),
    # ComboSpec(
    #     "9-2",
    #     {"D": 1.0, "E_v": 0.525, "E_h": 0.525, "L": 0.75, "S": 0.1},
    #     ("strength", "principal E"),
    #     DesignMethod.ASD,
    # ),
    ComboSpec(
        "9-2x",
        {"D": 1.0, "E_v": 0.525, "E_hx": 0.525, "L": 0.75, "S": 0.1},
        ("strength", "principal E"),
        DesignMethod.ASD,
    ),
    ComboSpec(
        "9-2y",
        {"D": 1.0, "E_v": 0.525, "E_hy": 0.525, "L": 0.75, "S": 0.1},
        ("strength", "principal E"),
        DesignMethod.ASD,
    ),
    # ComboSpec("10", {"D": 0.6, "E_v": -0.7, "E_h": 0.7}, ("strength", "principal E", "uplift"), DesignMethod.ASD),
    ComboSpec("10x", {"D": 0.6, "E_v": -0.7, "E_hx": 0.7}, ("strength", "principal E", "uplift"), DesignMethod.ASD),
    ComboSpec("10y", {"D": 0.6, "E_v": -0.7, "E_hy": 0.7}, ("strength", "principal E", "uplift"), DesignMethod.ASD),
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
    # "E_h": LoadCase.SEISMIC,
    "E_hx": LoadCase.SEISMIC,
    "E_hy": LoadCase.SEISMIC,
}


def create_base_model(
    design_method: DesignMethod = DesignMethod.LRFD,
    load_cases: LoadCase | list[LoadCase] | None = None,
    tags: str | None = None,
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

    base_model.add_section(
        "s_rigid",
        A=1e9,
        Iy=1e9,
        Iz=1e9,
        J=1e9,
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
            base_model.add_load_combo(spec.name, dict(spec.factors), combo_tags=list(spec.tags))
            continue
        if spec.design_method is not None and spec.design_method != design_method:
            continue
        if spec.service_flag is not None and not flags[spec.service_flag]:
            continue
        # Include only if every referenced load is available. Combo unchanged.
        if not all(_LABEL_TO_CASE[label] in available_cases for label in spec.factors):
            continue
        base_model.add_load_combo(spec.name, dict(spec.factors), combo_tags=list(spec.tags))

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

    getattr(node, "_load_metadata")["node_loads"].append({
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

    getattr(member, "_load_metadata")["point_loads"].append({
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

    getattr(member, "_load_metadata")["dist_loads"].append({
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


def add_section_from_shape(
    model: FEModel3D,
    shape: str,
    name: str | None = None,
    prefix: str = "s_",
    minor_axis_bending: bool = False,
) -> Section:
    """Create a section from a shape and add it to a PyNite model.

    Args:
        model: PyNite model.
        shape: Shape designation.
        name: Desired section name.
        prefix: Required section prefix.
        minor_axis_bending: If True, swap Iy and Iz so the section
            bends about its minor axis in the model.

    Returns:
        Created section object.
    """
    section = Section.from_shape(shape)

    if name:
        name = name.lower()
    else:
        name = section.designation.lower()

    if not name.startswith(prefix):
        name = f"{prefix}{name}"

    section.name = name
    iy = section.second_moment_of_area_y_axis
    iz = section.second_moment_of_area_x_axis

    if minor_axis_bending:
        iy, iz = iz, iy

    model.add_section(
        name,
        A=section.area,
        Iy=iy,
        Iz=iz,
        J=section.torsional_constant,
    )

    return section


def add_sections_from_shapes(
    model: FEModel3D,
    sections: dict[str, str],
) -> dict[str, Section]:
    """Add multiple sections from shapes.

    Args:
        model: PyNite model.
        sections: Mapping of section_name -> shape_name.

    Returns:
        Mapping of section_name -> Section.
    """
    output = {}

    for name, shape in sections.items():
        output[name] = add_section_from_shape(
            model=model,
            shape=shape,
            name=name,
        )

    return output


def add_mpc(
    model: FEModel3D,
    independent_node: str,
    dependent_nodes: list[str],
    material: str = "mat_rigid",
    section: str = "s_rigid",
) -> list[str]:
    """Create rigid MPC members.

    Args:
        model: PyNite model.
        primary_node: Primary MPC node.
        secondary_nodes: Secondary nodes connected to the MPC.
        material: Material name.
        section: Section name.

    Returns:
        List of ber names.
    """
    if not independent_node.startswith("n_mpc_"):
        raise ValueError(f"Primary node '{independent_node}' must start with 'n_mpc_'")

    suffix = independent_node.removeprefix("n_mpc_")

    member_names = []

    for i, node in enumerate(dependent_nodes, start=1):
        member_name = f"mpc_{suffix}_{i}"

        model.add_member(
            member_name,
            independent_node,
            node,
            material,
            section,
        )

        member_names.append(member_name)

    return member_names
