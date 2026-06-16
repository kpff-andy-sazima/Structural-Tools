from dataclasses import dataclass, field
from importlib import resources
import warnings

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
    building_period_coefficient: float = 0.02
    building_period_exponent: float = 0.75

    def __post_init__(self):
        if self.system_index not in DESIGN_COEFFICIENTS_TABLE_12_2_1.index:
            raise ValueError(
                f"'{self.system_index}' is not a valid Table 12.2-1 index. "
                f"Valid indices: {list(DESIGN_COEFFICIENTS_TABLE_12_2_1.index)}"
            )

        row = DESIGN_COEFFICIENTS_TABLE_12_2_1.loc[self.system_index]
        self.category = row["Category"]
        self.system = row["Seismic Force-Resisting System"]
        self.response_modification_coefficient = row["R"]
        self.overstrength_factor = row["Omega_0"]
        self.deflection_amplification_factor = row["C_d"]

    @property
    def c_t(self):
        return self.building_period_coefficient

    @property
    def x(self):
        return self.building_period_exponent

    @property
    def r(self):
        return self.response_modification_coefficient

    @property
    def omega_0(self):
        return self.overstrength_factor

    @property
    def c_d(self):
        return self.deflection_amplification_factor


@dataclass
class Level:
    height: float | Physical = 0
    weight: float | Physical = 0

    def __post_init__(self):
        if isinstance(self.height, Physical):
            self.height = float(self.height)
        if isinstance(self.weight, Physical):
            self.weight = float(self.weight)


@dataclass
class Structure:
    lateral_system_x: LateralSystem
    lateral_system_y: LateralSystem
    structural_height: float | Physical
    levels_input: dict[int, Level]
    vertical_system: VerticalSystem = field(default_factory=VerticalSystem)
    site_class: SiteClass = SiteClass.D
    risk_category: RiskCategory = RiskCategory.II

    def __post_init__(self):
        # Check for level numbers increasing consecutively
        expected = list(range(1, len(self.levels_input) + 1))
        if list(self.levels_input.keys()) != expected:
            raise ValueError(
                f"levels_data keys must be consecutive integers starting at 1, got {list(self.levels_input.keys())}"
            )

        # Set the structural height to a float
        if isinstance(self.structural_height, Physical):
            self.structural_height = float(self.structural_height)

        # Roof should have 0 height. Force it to be 0
        if self.levels_input[self.roof].height != 0:
            self.levels_input[self.roof].height = 0
            warnings.warn(f"Roof level ({self.roof}) height overridden to 0", stacklevel=2)

        # Level 1 should have 0 weight. Force it to be 0
        if self.levels_input[1].weight != 0:
            self.levels_input[1].weight = 0
            warnings.warn("Ground level (1) weight overridden to 0", stacklevel=2)

        self.period_x = self.lateral_system_x.c_t * self.structural_height**self.lateral_system_x.x
        self.period_y = self.lateral_system_y.c_t * self.structural_height**self.lateral_system_y.x

    @property
    def roof(self) -> int:
        return max(self.levels_input.keys())

    @property
    def floor_levels(self) -> list[int]:
        return [level for level in self.levels_input.keys() if level != self.roof]

    @property
    def levels(self) -> list[int]:
        return self.floor_levels + [self.roof]

    @property
    def diaphragm_levels(self) -> list[int]:
        return self.levels[1:]

    @property
    def shear_wall_levels(self) -> list[int]:
        return self.floor_levels

    @property
    def elevations(self) -> dict[int, float]:
        elevs = {}
        cumulative = 0.0
        for level, data in self.levels_input.items():
            elevs[level] = cumulative
            cumulative += data.height
        return elevs

    @property
    def effective_seismic_weight(self) -> float:
        return sum(float(level.weight) for level in self.levels_input.values())

    @property
    def levels_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "level height": {k: v.height for k, v in self.levels_input.items()},
                "level elevation": self.elevations,
                "level weight": {k: v.weight for k, v in self.levels_input.items()},
            },
            index=pd.Index(self.levels, name="Level"),
        )

    @property
    def lateral_systems_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "response modification coefficient": [self.lateral_system_x.r, self.lateral_system_y.r],
                "overstrength factor": [self.lateral_system_x.omega_0, self.lateral_system_y.omega_0],
                "deflection amplitude factor": [self.lateral_system_x.c_d, self.lateral_system_y.c_d],
            },
            index=pd.Index(["lateral system x", "lateral system y"], name="Lateral System"),
        )
