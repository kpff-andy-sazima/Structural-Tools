from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from importlib import resources

import pandas as pd

from .. import DesignMethod, LoadCase

NDS_LOAD_CASE_FACTOR_SEISMIC_ASD = 1 / 2.8
NDS_LOAD_CASE_FACTOR_WIND_ASD = 1 / 2
NDS_LOAD_CASE_FACTOR_SEISMIC_LRFD = 0.5
NDS_LOAD_CASE_FACTOR_WIND_LRFD = 0.8

SDPWS_TABLE_4_3_A = pd.read_csv(
    filepath_or_buffer=resources
    .files("structural_tools.data")
    .joinpath("sdpws_2021_table_4_3_a.csv")
    .open("r", encoding="utf-8"),
)

# SDPWS_TABLE_4_2_A = pd.read_csv(
#     filepath_or_buffer=resources.files("structural_tools.data")
#     .joinpath("sdpws_2021_table_4_2_a.csv")
#     .open("r", encoding="utf-8"),
# )

# SDPWS_TABLE_4_2_B = pd.read_csv(
#     filepath_or_buffer=resources.files("structural_tools.data")
#     .joinpath("sdpws_2021_table_4_2_b.csv")
#     .open("r", encoding="utf-8"),
# )

SDPWS_TABLE_4_2_C = pd.read_csv(
    filepath_or_buffer=resources
    .files("structural_tools.data")
    .joinpath("sdpws_2021_table_4_2_c.csv")
    .open("r", encoding="utf-8"),
)


class Nail(Enum):
    COMMON_6D = "6d common"
    COMMON_8D = "8d common"
    COMMON_10D = "10d common"
    GALV_CASING_6D = "6d galvanized casing"
    GALV_CASING_8D = "8d galvanized casing"
    GALV_ROOFING_11GA = "11 gauge galvanized roofing"


class SheathingApplication(Enum):
    DIAPHRAGM = auto()
    SHEAR_WALL = auto()


class SheathingMaterial(Enum):
    WSP_STRUCTURAL_I = "Wood Structural Panels - Structural I"
    WSP_SHEATHING = "Wood Structural Panels - Sheathing"
    PLYWOOD_SIDING = "Plywood Siding"
    PARTICLEBOARD_SHEATHING = "Particleboard Sheathing"
    SFS = "Structural Fiberboard Sheathing"


class PanelType(Enum):
    OSB = "OSB"
    PLY = "PLY"


VALID_NOMINAL_PANEL_THICKNESSES = {
    5 / 16,
    3 / 8,
    7 / 16,
    15 / 32,
    1 / 2,
    19 / 32,
    25 / 32,
}

VALID_NAIL_BEARING_LENGTHS = {
    1 + 1 / 4,
    1 + 3 / 8,
    1 + 1 / 2,
}

VALID_NAIL_SPACING = {
    2,
    3,
    4,
    6,
}


VALID_SHEATHED_SIDES = {
    1,
    2,
}

VALID_MINIMUM_NOMINAL_WIDTHS = {
    2,
    3,
}

VALID_CASES = {
    1,
    2,
    3,
    4,
    5,
    6,
}


@dataclass
class Sheathing:
    nail_spacing: int | None = None
    sheathing_material: SheathingMaterial | None = None
    minimum_nominal_panel_thickness: float | None = None
    minimum_nail_bearing_length: float | None = None
    nail: Nail | None = None
    panel_type: PanelType | None = PanelType.PLY
    minimum_nominal_width_of_nailed_face: int | None = None
    adjoining_panel_edge_location: Sequence[int] | None = None
    blocking: bool | None = False

    def __post_init__(self):
        # Value checks
        if (
            self.minimum_nominal_panel_thickness
            and self.minimum_nominal_panel_thickness not in VALID_NOMINAL_PANEL_THICKNESSES
        ):
            raise ValueError(
                f"Invalid minimum nominal panel thickness: {self.minimum_nominal_panel_thickness}.\nValid values include: {VALID_NOMINAL_PANEL_THICKNESSES}"
            )
        if self.minimum_nail_bearing_length and self.minimum_nail_bearing_length not in VALID_NAIL_BEARING_LENGTHS:
            raise ValueError(
                f"Invalid minimum nail bearing length: {self.minimum_nail_bearing_length}.\nValid values include: {VALID_NAIL_BEARING_LENGTHS}"
            )
        if self.nail_spacing and self.nail_spacing not in VALID_NAIL_SPACING:
            raise ValueError(f"Invalid nail spacing: {self.nail_spacing}.\nValid values include: {VALID_NAIL_SPACING}")
        if (
            self.minimum_nominal_width_of_nailed_face
            and self.minimum_nominal_width_of_nailed_face not in VALID_MINIMUM_NOMINAL_WIDTHS
        ):
            raise ValueError(
                f"Invalid minimum nominla width of nailed face: {self.minimum_nominal_width_of_nailed_face}.\nValid values include: {VALID_MINIMUM_NOMINAL_WIDTHS}"
            )
        if self.adjoining_panel_edge_location:
            invalid_cases = set(self.adjoining_panel_edge_location) - set(VALID_CASES)
            if invalid_cases:
                raise ValueError(
                    f"Invalid adjoining panel edge location case(s): {sorted(invalid_cases)}.\n"
                    f"Valid cases include: {sorted(VALID_CASES)}"
                )


def _apply_filter(df: pd.DataFrame, filter_dataclass) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)

    for field, value in filter_dataclass.__dict__.items():
        if value is None or field == "sheathed_sides":
            continue
        elif field == "sheathing_material":
            mask &= df["Sheathing Material"] == value.value
        elif field == "minimum_nominal_panel_thickness":
            mask &= df["Minimum Nominal Panel Thickness"] == value
        elif field == "minimum_nail_bearing_length":
            mask &= df["Minimum Nail Bearing Length in Framing Member or Blocking"] == value
        elif field == "nail":
            mask &= df["Nail Type and Size"] == value.value
        elif field == "nail_spacing":
            mask &= df["Panel Edge Nail Spacing"] == value
        elif field == "panel_type":
            mask &= df["Panel Type"] == value.value
        elif field == "minimum_nominal_width_of_nailed_face":
            mask &= df["Minimum Nominal Width of Nailed Face at Adjoining Panel Edges and Boundaries"] == value
        elif field == "adjoining_panel_edge_location":
            requested = {str(c) for c in value}
            mask &= df["Case"].astype(str).apply(lambda s: requested.issubset(set(s.replace(",", "").replace(" ", ""))))

    return df[mask]


def get_viable_sheathing(
    demand: float,
    sheathing: Sheathing,
    sheathed_sides: int = 1,
    load_case: LoadCase = LoadCase.SEISMIC,
    design_method: DesignMethod = DesignMethod.ASD,
    sheathing_application: SheathingApplication = SheathingApplication.SHEAR_WALL,
) -> pd.DataFrame:
    if sheathed_sides not in VALID_SHEATHED_SIDES:
        raise ValueError(
            f"Invalid number of sheathed sides: {sheathed_sides}.\nValid values include: {VALID_SHEATHED_SIDES}"
        )
    # Filter based on sheathing parameters
    if sheathing_application == SheathingApplication.SHEAR_WALL:
        filtered_sheathing_options: pd.DataFrame = _apply_filter(SDPWS_TABLE_4_3_A, filter_dataclass=sheathing)
    else:
        # TODO: Add blocking. Currently just pull unblocked values
        if sheathing.blocking:
            # filtered_sheathing_options: pd.DataFrame = _apply_filter(SDPWS_TABLE_4_2_A, filter_dataclass=sheathing)
            filtered_sheathing_options: pd.DataFrame = _apply_filter(SDPWS_TABLE_4_2_C, filter_dataclass=sheathing)
            pass
        else:
            filtered_sheathing_options: pd.DataFrame = _apply_filter(SDPWS_TABLE_4_2_C, filter_dataclass=sheathing)

    # Scale by load case
    if design_method == DesignMethod.ASD:
        if load_case == LoadCase.SEISMIC:
            load_case_factor = NDS_LOAD_CASE_FACTOR_SEISMIC_ASD
        else:
            load_case_factor = NDS_LOAD_CASE_FACTOR_WIND_ASD
    else:
        if load_case == LoadCase.SEISMIC:
            load_case_factor = NDS_LOAD_CASE_FACTOR_SEISMIC_LRFD
        else:
            load_case_factor = NDS_LOAD_CASE_FACTOR_WIND_LRFD

    # Scale by 1 or 2 based on number of sheathed sides and load case
    filtered_sheathing_options["Adjusted Unit Shear"] = (
        filtered_sheathing_options["Unit Shear"] * float(sheathed_sides) * load_case_factor
    )

    # Filter based on demand
    possible_sheathing_options = filtered_sheathing_options[filtered_sheathing_options["Adjusted Unit Shear"] >= demand]

    # If no options fulfill the demand requirement, return the next highest one for the engineer to look at later
    # Otherwise, return the possible sheathing options
    if possible_sheathing_options.empty:
        return filtered_sheathing_options.loc[[filtered_sheathing_options["Adjusted Unit Shear"].idxmax()]]
    else:
        return possible_sheathing_options


def select_sheathing(
    viable_sheathing: pd.DataFrame,
    criteria_column: str = "Adjusted Unit Shear",
    criteria: str = "min",
    return_columns: list[str] = [
        "Adjusted Unit Shear",
        "Panel Edge Nail Spacing",
        "Apparent Shear Stiffness Value",
    ],
):
    if criteria not in ["max", "min"]:
        raise ValueError(f"{criteria} not a valid criteria. Valid criteria are 'max' and 'min'")

    # We already handle the case where demand is too high in the get_viable_sheathing() function, but this is just a backup
    if viable_sheathing.empty:
        return pd.Series({
            "Adjusted Unit Shear": 0.001,
            "Panel Edge Nail Spacing": 0,
            "Apparent Shear Stiffness Value": 0.001,
        })

    selectors = {
        "max": pd.Series.idxmax,
        "min": pd.Series.idxmin,
    }

    idx = selectors[criteria](viable_sheathing[criteria_column])
    return viable_sheathing.loc[idx, return_columns]


# Get sheathing properties
def get_sheathing_properties(
    dataframe: pd.DataFrame,
    sheathing: Sheathing,
    criteria_column: str = "Adjusted Unit Shear",
    criteria: str = "min",
    sheathing_application: SheathingApplication = SheathingApplication.SHEAR_WALL,
) -> pd.DataFrame:
    """For each wall demand and number of sheathed sides in a dataframe, get back the dataframe with each wall sheathing option based on the criteria given.

    Args:
        dataframe (pd.DataFrame): DataFrame with 'adjusted unit shear demand' and 'sheathed sides' columns
        sheathing (Sheathing): sheathing properties class
        criteria_column (str, optional): The column that is used to choose the sheathing option. Defaults to "Panel Edge Nail Spacing".
        criteria (str, optional): The criteria that decides the option using the criteria_column. Defaults to "max".

    Raises:
        ValueError: If the dataframe argument does not have 'adjusted unit shear demand' and/or 'sheathed sides'.

    Returns:
        pd.DataFrame: DataFrame with the columns ['adjusted unit shear capacity', 'nail spacing', 'shear stiffness', 'shear dcr']
    """
    # Shear wall sheathing selection needs demand and number of sheathed sides as 2 sheathed sides doubles capacity
    if sheathing_application == SheathingApplication.SHEAR_WALL:
        required = {"adjusted unit shear demand", "sheathed sides"}
        if not required.issubset(dataframe.columns):
            raise ValueError(
                "The given DataFrame requires the following columns: 'adjusted unit shear demand', 'sheathed sides'"
            )

        properties: pd.DataFrame = pd.DataFrame()
        properties[["adjusted unit shear capacity", "nail spacing", "shear stiffness"]] = dataframe[
            ["adjusted unit shear demand", "sheathed sides"]
        ].apply(
            lambda row: select_sheathing(
                viable_sheathing=get_viable_sheathing(
                    demand=row["adjusted unit shear demand"],
                    sheathing=sheathing,
                    sheathed_sides=row["sheathed sides"],
                    sheathing_application=sheathing_application,
                ),
                criteria_column=criteria_column,
                criteria=criteria,
            ),
            axis=1,
        )
    # Diaphragm sheathing selection only needs the demand as it is only ever sheathed on top
    else:
        # required = {"adjusted unit shear demand"}
        if isinstance(dataframe, pd.Series):
            dataframe = dataframe.to_frame()

        # Guarantee a single column
        if dataframe.shape[1] != 1:
            raise ValueError("The given DataFrame requires only one column for the demands (with any name)")

        # Ensure the column is named correctly
        dataframe.columns = ["adjusted unit shear demand"]

        properties: pd.DataFrame = pd.DataFrame()
        properties[["adjusted unit shear capacity", "shear stiffness"]] = dataframe.apply(
            lambda row: select_sheathing(
                viable_sheathing=get_viable_sheathing(
                    demand=row["adjusted unit shear demand"],
                    sheathing=sheathing,
                    sheathed_sides=1,
                    sheathing_application=sheathing_application,
                ),
                criteria_column=criteria_column,
                criteria=criteria,
                return_columns=["Adjusted Unit Shear", "Apparent Shear Stiffness Value"],
            ),
            axis=1,
        )
    properties["shear dcr"] = dataframe["adjusted unit shear demand"] / properties["adjusted unit shear capacity"]
    return properties
