from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.structural_tools.wood.sheathing import Sheathing

from ..structure import Structure
from ..typing import FloatLike
from .constants import (
    SPDWS_LOAD_CASE_FACTOR_SEISMIC_ASD,
    # SPDWS_LOAD_CASE_FACTOR_SEISMIC_LRFD,
    # SPDWS_LOAD_CASE_FACTOR_WIND_ASD,
    # SPDWS_LOAD_CASE_FACTOR_WIND_LRFD,
)
from .sheathing import get_sheathing_properties


def create_shear_walls_dataframe_from_csv(
    shear_wall_input_csv: Path | str,
    structure: Structure,
    tributary_area_check: dict[int, float] | float | None = None,
) -> pd.DataFrame:
    """Generate the shear wall dataframe with the following assumptions:

    - Wood shear walls
    - All stacking shear walls

    If walls don't stack, you can edit the dataframe after it is created.

    Args:
        shear_wall_input_csv (Path | str): CSV with shear wall info
        structure (Structure): Structure object created for seismic loading creation
        tributary_area_check (list[float] | float | None): list or single float for checking that all trib areas add up to the total for a level

    Returns:
        pd.DataFrame: Shear wall dataframe
    """
    shear_walls = pd.read_csv(shear_wall_input_csv)
    shear_walls.columns = shear_walls.columns.str.lower()

    shear_walls = pd.concat(
        [shear_walls.assign(level=level) for level in structure.shear_wall_levels], ignore_index=True
    )

    # Ensure a line column exists and populate it
    derived_line = shear_walls["wall"].str.split("--", n=1).str[0].str.strip()
    if "line" not in shear_walls.columns:
        shear_walls["line"] = derived_line
    else:
        mask = shear_walls["line"].isna() | (shear_walls["line"].astype(str).str.strip() == "")
        shear_walls.loc[mask, "line"] = derived_line[mask]

    # Fill empty sheathed sides with 1
    if "sheathed sides" not in shear_walls.columns:
        shear_walls["sheathed sides"] = 1
    else:
        shear_walls["sheathed sides"] = shear_walls["sheathed sides"].fillna(1)

    # Fill empties with 0
    shear_walls = shear_walls.fillna(0)

    # Index columns set
    shear_walls = shear_walls.rename(columns={"level": "Level", "wall": "Wall", "direction": "Direction"})
    shear_walls = shear_walls.set_index(["Level", "Wall"])

    # Add trib area automatically. Drop the trib width and trib length columns after area is determined
    if "tributary area" not in shear_walls.columns:
        shear_walls["tributary area"] = 0

    shear_walls["tributary area"] = shear_walls["tributary area"].where(
        shear_walls["tributary area"] == 0, shear_walls["tributary width"] * shear_walls["tributary length"]
    )
    shear_walls.drop(columns=["tributary length", "tributary width"], inplace=True)

    total_trib_area = None
    if tributary_area_check:
        total_trib_area = check_shear_wall_tributary_area(
            shear_walls_dataframe=shear_walls, tributary_area_check=tributary_area_check
        )

    shear_walls["wall height"] = shear_walls.index.get_level_values("Level").map(structure.levels_data["level height"])
    if total_trib_area:
        weight_above = structure.levels_data["level weight"].shift(-1).fillna(0)
        weight_per_area = weight_above / total_trib_area
        shear_walls["level area weight"] = shear_walls.index.get_level_values("Level").map(weight_per_area)

    return shear_walls


def check_shear_wall_tributary_area(
    shear_walls_dataframe: pd.DataFrame,
    tributary_area_check: dict[int, float] | FloatLike,
) -> pd.DataFrame:
    total_trib_area = shear_walls_dataframe.groupby(level="Level")["tributary area"].sum()

    # Convert a single value into a dict for all levels
    if isinstance(tributary_area_check, FloatLike):
        tributary_area_check = {level: float(tributary_area_check) for level in total_trib_area.index}

    # Check that every level in the wall dataframe has a check value
    missing_levels = set(total_trib_area.index) - set(tributary_area_check)
    extra_levels = set(tributary_area_check) - set(total_trib_area.index)

    if missing_levels:
        raise ValueError(f"Trib area check missing levels: {sorted(missing_levels)}")

    if extra_levels:
        raise ValueError(f"Trib area check contains extra levels: {sorted(extra_levels)}")

    failures: list[str] = []

    for level, actual_area in total_trib_area.items():
        expected_area = tributary_area_check[level]

        if not math.isclose(
            actual_area,
            expected_area,
            rel_tol=1e-3,
            abs_tol=1e-1,
        ):
            failures.append(
                f"Level {level}: expected {expected_area:.3f}, got {actual_area:.3f}\nEnter manual tributary areas (or updated tributary widths and tributary lengths) in the input CSV file to match the correct total area."
            )

    if failures:
        raise ValueError("Trib area check failed:\n" + "\n".join(failures))

    return total_trib_area


def assign_values_for_all_levels_per_wall(
    shear_walls_dataframe: pd.DataFrame,
    mapping_dict: dict[str, dict[str, FloatLike]],
) -> pd.DataFrame:
    """Assign values to columns for a specific wall on all levels.

    Args:
        shear_walls_dataframe (pd.DataFrame): Shear walls dataframe created by "create_shear_walls_dataframe_from_csv"
        mapping_dict (dict[str, dict[str, FloatLike]]): Dictionary of values to map to walls. No levels will be specified as all will match. It should look like:
            mapping_dict = {"wall": {"dataframe column": value}}
    Returns:
        pd.DataFrame: Updated dataframe
    """
    for wall, properties in mapping_dict.items():
        for column, value in properties.items():
            shear_walls_dataframe.loc[
                shear_walls_dataframe.index.get_level_values("Wall") == wall,
                column,
            ] = value
    return shear_walls_dataframe


def assign_values_for_all_walls_per_level(
    shear_walls_dataframe: pd.DataFrame,
    mapping_dict: dict[int, dict[str, FloatLike]],
) -> pd.DataFrame:
    """Assign values to columns for all walls on a specific level.

    Args:
        shear_walls_dataframe (pd.DataFrame): Shear walls dataframe created by "create_shear_walls_dataframe_from_csv"
        mapping_dict (dict[str, dict[str, FloatLike]]): Dictionary of values to map to walls. No levels will be specified as all will match. It should look like:
            mapping_dict = {1: {"dataframe column": value}}
    Returns:
        pd.DataFrame: Updated dataframe
    """
    for level, properties in mapping_dict.items():
        for column, value in properties.items():
            shear_walls_dataframe.loc[
                shear_walls_dataframe.index.get_level_values("Level") == level,
                column,
            ] = value
    return shear_walls_dataframe


def calculate_flexible_seismic_demand(shear_walls_dataframe):
    necessary_columns = ["line", "tributary area", "wall length", "tributary weight"]
    if necessary_columns not in shear_walls_dataframe.columns:
        raise ValueError(f"The supplied dataframe is missing one of the following columns:\n{necessary_columns}")

    # Automatically extract the shear line by splitting at "_" and taking the first section
    shear_walls_dataframe["line"] = shear_walls_dataframe.index.get_level_values("Wall").str.split("--").str[0]

    # Get tributary area and wall length for the shear line
    shear_walls_dataframe["line tributary area"] = shear_walls_dataframe.groupby(["Level", "line"])[
        "tributary area"
    ].transform("sum")
    shear_walls_dataframe["line wall length"] = shear_walls_dataframe.groupby(["Level", "line"])[
        "wall length"
    ].transform("sum")

    shear_walls_dataframe["line shear demand"] = (
        shear_walls_dataframe["line tributary area"] * shear_walls_dataframe["tributary weight"]
    )
    shear_walls_dataframe["line unit shear"] = (
        shear_walls_dataframe["line shear demand"] / shear_walls_dataframe["line wall length"]
    )
    shear_walls_dataframe["line cumulative shear demand"] = shear_walls_dataframe.groupby(level="Wall")[
        "line shear demand"
    ].cumsum()
    shear_walls_dataframe["line cumulative unit shear"] = (
        shear_walls_dataframe["line cumulative shear demand"] / shear_walls_dataframe["line wall length"]
    )

    shear_walls_dataframe["shear demand"] = (
        shear_walls_dataframe["line unit shear"] * shear_walls_dataframe["wall length"]
    )
    shear_walls_dataframe["cumulative shear demand"] = shear_walls_dataframe.groupby(level="Wall")[
        "shear demand"
    ].cumsum()
    shear_walls_dataframe["unit shear demand"] = (
        shear_walls_dataframe["cumulative shear demand"] / shear_walls_dataframe["wall length"]
    )
    shear_walls_dataframe["adjusted unit shear demand"] = (
        shear_walls_dataframe["unit shear demand"] * SPDWS_LOAD_CASE_FACTOR_SEISMIC_ASD
    )

    shear_walls_dataframe["aspect ratio"] = shear_walls_dataframe["wall height"] / shear_walls_dataframe["wall length"]

    return shear_walls_dataframe


def choose_shear_wall_sheathing(shear_walls_dataframe: pd.DataFrame, sheathing: Sheathing, dcr_check_value: float = 1):
    # TODO need to check that the necessary columns are contained in the DF
    shear_walls_dataframe[["adjusted unit shear capacity", "nail spacing", "shear stiffness", "shear dcr"]] = (
        get_sheathing_properties(
            shear_walls_dataframe[["adjusted unit shear demand", "sheathed sides"]], sheathing=sheathing
        )
    )

    # Assign 2-sided sheathing to walls requiring it, then recalc capacity for just those walls (more efficient than passing the full wall dataframe)
    if (shear_walls_dataframe["shear dcr"] > dcr_check_value).any():
        shear_walls_dataframe.loc[shear_walls_dataframe["shear dcr"] > dcr_check_value, "sheathed sides"] = 2
        shear_walls_dataframe.loc[
            shear_walls_dataframe["shear dcr"] > dcr_check_value,
            ["adjusted unit shear capacity", "nail spacing", "shear stiffness", "shear dcr"],
        ] = get_sheathing_properties(
            shear_walls_dataframe.loc[
                shear_walls_dataframe["shear dcr"] > dcr_check_value,
                ["adjusted unit shear demand", "sheathed sides"],
            ],
            sheathing=sheathing,
        )

    return shear_walls_dataframe


def design_shear_walls_flexible(shear_walls_dataframe: pd.DataFrame, sheathing: Sheathing, dcr_check_value: float = 1):
    shear_walls_dataframe = calculate_flexible_seismic_demand(shear_walls_dataframe)
    return choose_shear_wall_sheathing(shear_walls_dataframe, sheathing, dcr_check_value)
