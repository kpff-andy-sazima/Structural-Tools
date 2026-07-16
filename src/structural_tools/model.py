"""Modeling tools"""

from Pynite import FEModel3D

from .asce import DesignMethod

from copy import deepcopy
from dataclasses import dataclass


@dataclass
class ModelMaterial:
    name: str  # Material name
    E: float  # Modulus of elasticity
    nu: float  # Poisson's ratio
    rho: float  # Density
    fy: float  # Yield stress

    def __post_init__(self):
        G: float = self.E / (2 * (1 + self.nu))  # Shear modulus of elasticity


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
    include_ossc_service_combos: bool = True,
    include_ossc_creep_combos: bool = True,
    include_asce_service_combos: bool = False,
    include_asce_creep_combos: bool = False,
    materials: ModelMaterial | list[ModelMaterial] = MATERIAL_STEEL_50_KSI_KIP_INCH,
) -> FEModel3D:
    base_model = FEModel3D()

    # Wood
    E = 1700  # Modulus of elasticity (ksi)
    Fy = 2.400  # Yield stress (ksi)
    nu = 0.3  # Poisson's ratio
    G = E / (2 * (1 + nu))  # Shear modulus of elasticity (ksi)
    rho = 33 / (12**3) / 1000  # Density (kci)
    _ = base_model.add_material(name="glulam", E=E, G=G, nu=nu, rho=rho, fy=Fy)

    if not isinstance(list, materials):
        materials = [materials]

    for material in materials:
        base_model.add_material(
            name=material.name, E=material.E, G=material.G, nu=material.nu, rho=material.rho, fy=material.fy
        )

    # Load combinations
    # LRFD combos
    if design_method == DesignMethod.LRFD:
        base_model.add_load_combo("1a", {"D": 1.4}, ["strength", "principal D"])
        base_model.add_load_combo("2a-1", {"D": 1.2, "L": 1.6, "L_r": 0.5}, ["strength", "principal L"])
        base_model.add_load_combo("2a-2", {"D": 1.2, "L": 1.6, "S": 0.3}, ["strength", "principal L"])
        base_model.add_load_combo("2a-3", {"D": 1.2, "L": 1.6, "R": 0.5}, ["strength", "principal L"])
        base_model.add_load_combo("3a-1", {"D": 1.2, "L_r": 1.6, "L": 1.0}, ["strength", "principal L_r"])
        base_model.add_load_combo("3a-2", {"D": 1.2, "S": 1.0, "L": 1.0}, ["strength", "principal S"])
        base_model.add_load_combo("3a-3", {"D": 1.2, "R": 1.6, "L": 1.0}, ["strength", "principal R"])
        base_model.add_load_combo("3a-4", {"D": 1.2, "L_r": 1.6, "W": 0.5}, ["strength", "principal L_r"])
        base_model.add_load_combo("3a-5", {"D": 1.2, "S": 1.0, "W": 0.5}, ["strength", "principal S"])
        base_model.add_load_combo("3a-6", {"D": 1.2, "R": 1.6, "W": 0.5}, ["strength", "principal R"])
        base_model.add_load_combo("4a-1", {"D": 1.2, "W": 1.0, "L": 1.0, "L_r": 0.5}, ["strength", "principal W"])
        base_model.add_load_combo("4a-2", {"D": 1.2, "W": 1.0, "L": 1.0, "S": 0.3}, ["strength", "principal W"])
        base_model.add_load_combo("4a-3", {"D": 1.2, "W": 1.0, "L": 1.0, "R": 0.5}, ["strength", "principal W"])
        base_model.add_load_combo("5a", {"D": 0.9, "W": -1.0}, ["strength", "principal W"])
        base_model.add_load_combo(
            "6", {"D": 1.2, "E_v": 1.0, "E_h": 1.0, "L": 1.0, "S": 0.15}, ["strength", "principal E"]
        )
        base_model.add_load_combo("7", {"D": 0.9, "E_v": -1.0, "E_h": 1.0}, ["strength", "principal E"])
    else:
        base_model.add_load_combo("1a", {"D": 1.0}, ["strength", "principal D"])
        base_model.add_load_combo("2a", {"D": 1.0, "L": 1.0}, ["strength", "principal L"])
        base_model.add_load_combo("3a-1", {"D": 1.0, "L_r": 1.0}, ["strength", "principal L_r"])
        base_model.add_load_combo("3a-2", {"D": 1.0, "S": 0.7}, ["strength", "principal S"])
        base_model.add_load_combo("3a-3", {"D": 1.0, "R": 1.0}, ["strength", "principal R"])
        base_model.add_load_combo("4a-1", {"D": 1.0, "L": 0.75, "L_r": 0.75}, ["strength", "principal L"])
        base_model.add_load_combo("4a-2", {"D": 1.0, "L": 0.75, "S": 0.525}, ["strength", "principal L"])
        base_model.add_load_combo("4a-3", {"D": 1.0, "L": 0.75, "R": 0.75}, ["strength", "principal L"])
        base_model.add_load_combo("5a", {"D": 1.0, "W": 0.6}, ["strength", "principal W"])
        base_model.add_load_combo("6a-1", {"D": 1.0, "L": 0.75, "W": 0.45, "L_r": 0.75}, ["strength", "principal W"])
        base_model.add_load_combo("6a-2", {"D": 1.0, "L": 0.75, "W": 0.45, "S": 0.525}, ["strength", "principal W"])
        base_model.add_load_combo("6a-3", {"D": 1.0, "L": 0.75, "W": 0.45, "R": 0.75}, ["strength", "principal W"])
        base_model.add_load_combo("7a", {"D": 0.6, "W": -0.6}, ["strength", "principal W"])
        base_model.add_load_combo("8", {"D": 1.0, "E_v": 0.7, "E_h": 0.7}, ["strength", "principal E"])
        base_model.add_load_combo(
            "9", {"D": 1.0, "E_v": 0.525, "E_h": 0.525, "L": 0.75, "S": 0.1}, ["strength", "principal E"]
        )
        base_model.add_load_combo("10", {"D": 0.6, "E_v": -0.7, "E_h": 0.7}, ["strength", "principal E"])

    # Add service load combos
    if include_ossc_service_combos:
        base_model.add_load_combo("L", {"L": 1.0}, ["service"])
        base_model.add_load_combo("L_r", {"L_r": 1.0}, ["service"])
        base_model.add_load_combo("S", {"S": 1.0}, ["service"])
        base_model.add_load_combo("W", {"W": 1.0}, ["service"])
    if include_ossc_creep_combos:
        base_model.add_load_combo("0.5D+L", {"D": 0.5, "L": 1.0}, ["creep OSSC25 dry wood"])
        base_model.add_load_combo("D+L", {"D": 1.0, "L": 1.0}, ["creep OSSC25 moist wood"])
    if include_asce_service_combos:
        base_model.add_load_combo("CC.2-1a", {"D": 1.0, "L": 1.0}, ["service ASCE7-22"])
        base_model.add_load_combo("CC.2-1b", {"D": 1.0, "S_ser": 1.0}, ["service ASCE7-22"])
    if include_asce_creep_combos:
        base_model.add_load_combo("CC.2-2", {"D": 1.0, "L": 0.5}, ["creep ASCE7-22"])

    # Other load combos
    base_model.add_load_combo("D", {"D": 1.0}, ["self weight"])
    base_model.add_load_combo(
        "all", {"D": 1.0, "L": 1.0, "L_r": 1.0, "S": 1.0, "W": 1.0, "R": 1.0, "E_v": 1.0, "E_h": 1.0}, ["all nominal"]
    )

    return deepcopy(base_model)
