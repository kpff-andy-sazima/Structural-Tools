import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources

import pandas as pd

from .asce import CodeVersion, RiskCategory, SiteClass
from .typing import FloatLike

_TABLE_12_2_1_FILES: dict[CodeVersion, str] = {
    CodeVersion.ASCE_7_16: "asce_7_16_table_12_2_1.csv",
    CodeVersion.ASCE_7_22: "asce_7_22_table_12_2_1.csv",
}


@lru_cache(maxsize=None)
def design_coefficients_table(code_version: CodeVersion) -> pd.DataFrame:
    """Load ASCE 7 Table 12.2-1 for a given code version.

    Args:
        code_version: ASCE 7 edition to load.
    Returns:
        Table 12.2-1 indexed by system index.
    Raises:
        ValueError: If no table is registered for `code_version`.
    """
    try:
        filename = _TABLE_12_2_1_FILES[code_version]
    except KeyError:
        valid = ", ".join(version.name for version in _TABLE_12_2_1_FILES)
        raise ValueError(f"No Table 12.2-1 data for {code_version}. Available: {valid}.") from None
    source = resources.files("structural_tools.data").joinpath(filename)
    with source.open("r", encoding="utf-8") as file:
        return pd.read_csv(file).set_index("Index")


@dataclass
class VerticalSystem:
    system: str = "Default vertical system"


@dataclass
class LateralSystem:
    system_index: str
    building_period_coefficient: float = 0.02
    building_period_exponent: float = 0.75

    def __post_init__(self):
        self.system_index = self._normalize_index(self.system_index)

    @staticmethod
    def _normalize_index(system_index: str) -> str:
        """Normalize a Table 12.2-1 index to stripped uppercase.

        Args:
            system_index: Raw index as supplied by the caller.
        Returns:
            The index with surrounding whitespace removed and uppercased.
        """
        return str(system_index).strip().upper()

    def bind_code_version(self, code_version: CodeVersion) -> None:
        """Resolve Table 12.2-1 coefficients against a code version.

        Args:
            code_version: ASCE 7 edition supplied by the owning Structure.
        Raises:
            ValueError: If `system_index` is not in the table.
        """
        # Re-normalize in case system_index was reassigned after construction
        self.system_index = self._normalize_index(self.system_index)
        table = design_coefficients_table(code_version)
        if self.system_index not in table.index:
            raise ValueError(
                f"'{self.system_index}' is not a valid {code_version.name} Table 12.2-1 index. "
                f"Valid indices: {list(table.index)}"
            )
        row = table.loc[self.system_index]
        self.asce_code_version = code_version
        self.category = row["Category"]
        self.system = row["Seismic Force-Resisting System"]
        self.response_modification_coefficient = row["R"]
        self.overstrength_factor = row["Omega_0"]
        self.deflection_amplification_factor = row["C_d"]

    def _require(self, name: str):
        """Return a table-derived attribute, erroring if unbound.

        Args:
            name: Attribute populated by `bind_code_version`.
        Returns:
            The attribute value.
        Raises:
            RuntimeError: If the system has not been bound to a code version.
        """
        try:
            return getattr(self, name)
        except AttributeError:
            raise RuntimeError(
                f"{type(self).__name__}('{self.system_index}') has no code version yet. "
                "Assign it to a Structure or call bind_code_version() first."
            ) from None

    @property
    def c_t(self):
        return self.building_period_coefficient

    @property
    def x(self):
        return self.building_period_exponent

    @property
    def r(self):
        return self._require("response_modification_coefficient")

    @property
    def omega_0(self):
        return self._require("overstrength_factor")

    @property
    def c_d(self):
        return self._require("deflection_amplification_factor")


@dataclass
class Level:
    height: FloatLike = 0
    weight: FloatLike = 0

    def __post_init__(self):
        self.height = float(self.height)
        self.weight = float(self.weight)


@dataclass
class Structure:
    lateral_system_x: LateralSystem
    lateral_system_y: LateralSystem
    levels_input: dict[int, Level]
    structural_height: FloatLike | None = None
    vertical_system: VerticalSystem = field(default_factory=VerticalSystem)
    site_class: SiteClass = SiteClass.D
    risk_category: RiskCategory = RiskCategory.II
    asce_code_version: CodeVersion = CodeVersion.ASCE_7_22

    def __post_init__(self):
        # Single source of truth for the code version
        self.lateral_system_x.bind_code_version(self.asce_code_version)
        self.lateral_system_y.bind_code_version(self.asce_code_version)

        # Check for level numbers increasing consecutively
        expected = list(range(1, len(self.levels_input) + 1))
        if list(self.levels_input.keys()) != expected:
            raise ValueError(
                f"levels_data keys must be consecutive integers starting at 1, got {list(self.levels_input.keys())}"
            )
        # Roof should have 0 height. Force it to be 0
        if self.levels_input[self.roof].height != 0:
            self.levels_input[self.roof].height = 0
            warnings.warn(f"Roof level ({self.roof}) height overridden to 0", stacklevel=2)

        # Level 1 should have 0 weight. Force it to be 0
        if self.levels_input[1].weight != 0:
            self.levels_input[1].weight = 0
            warnings.warn("Ground level (1) weight overridden to 0", stacklevel=2)

        if self.structural_height:
            self.structural_height = float(self.structural_height)
        else:
            self.structural_height = sum(level.height for level in self.levels_input.values())

        self.period_x = self.lateral_system_x.c_t * self.structural_height**self.lateral_system_x.x
        self.period_y = self.lateral_system_y.c_t * self.structural_height**self.lateral_system_y.x
        self.t_x = self.period_x
        self.t_y = self.period_y

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
                "response modification coefficient": [
                    self.lateral_system_x.r,
                    self.lateral_system_y.r,
                ],
                "overstrength factor": [
                    self.lateral_system_x.omega_0,
                    self.lateral_system_y.omega_0,
                ],
                "deflection amplitude factor": [
                    self.lateral_system_x.c_d,
                    self.lateral_system_y.c_d,
                ],
            },
            index=pd.Index(["lateral system x", "lateral system y"], name="Lateral System"),
        )
