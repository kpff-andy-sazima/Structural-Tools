from dataclasses import dataclass
from importlib import resources

import pandas as pd
from forallpeople import Physical

from structural_tools.seismic.asce.parameters import RiskCategory, SiteClass

DESIGN_COEFFICIENTS_TABLE_12_2_1 = pd.read_csv(
    filepath_or_buffer=resources
    .files("structural_tools.data")
    .joinpath("asce_7_22_table_12_2_1.csv")
    .open("r", encoding="utf-8"),
).set_index("Index")


@dataclass
class VerticalSystem:
    system: str = "Default vertical system"


@dataclass
class LateralSystem:
    system_index: str

    def __post_init__(self):
        if self.system_index not in DESIGN_COEFFICIENTS_TABLE_12_2_1.index:
            raise ValueError(
                f"'{self.system_index}' is not a valid Table 12.2-1 index. "
                f"Valid indices: {list(DESIGN_COEFFICIENTS_TABLE_12_2_1.index)}"
            )

        row = DESIGN_COEFFICIENTS_TABLE_12_2_1.loc[self.system_index]
        self.category = row["Category"]
        self.system = row["Seismic Force-Resisting System"]
        self.r = row["R"]
        self.response_modification_coefficient = self.r
        self.omega_0 = row["Omega_0"]
        self.overstrength_factor = self.omega_0
        self.c_d = row["C_d"]
        self.deflection_amplification_factor = self.c_d


@dataclass
class Structure:
    lateral_system_x: LateralSystem
    lateral_system_y: LateralSystem
    structural_height: float
    vertical_system: VerticalSystem = VerticalSystem()
    number_of_levels: int = 1
    site_class: SiteClass = SiteClass.D
    risk_category: RiskCategory = RiskCategory.II

    def __post_init__(self):
        pass

    @property
    def levels(self) -> list[int]:
        return list(range(1, self.number_of_levels + 1))

    @property
    def roof(self) -> int:
        return self.number_of_levels + 1

    @property
    def diaphragm_levels(self) -> list[int]:
        return self.levels[1:] + [self.roof]

    @property
    def shear_wall_levels(self) -> list[int]:
        return self.levels
