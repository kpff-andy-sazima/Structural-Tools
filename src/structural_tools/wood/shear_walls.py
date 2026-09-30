from __future__ import annotations

import math
from pathlib import Path
from string import ascii_uppercase
from typing import Literal

import numpy as np
import pandas as pd

from ..asce.equivalent_lateral_force_procedure import SeismicLoads
from ..asce.seismic_parameters import (
    ASCE_ASD_8_D,
    ASCE_ASD_8_E_H,
    ASCE_ASD_8_E_V,
    ASCE_ASD_9_D,
    ASCE_ASD_9_E_H,
    ASCE_ASD_9_E_V,
    ASCE_ASD_9_L,
    ASCE_ASD_9_S,
    ASCE_ASD_10_D,
    ASCE_ASD_10_E_H,
    ASCE_ASD_10_E_V,
)
from ..conversions import KIPS_TO_LBS
from ..typing import FloatLike
from .constants import SPDWS_LOAD_CASE_FACTOR_SEISMIC_ASD
from .sheathing import Sheathing, get_sheathing_properties


def _debug_save(
    df: pd.DataFrame,
    filename: str,
    enabled: bool,
) -> None:
    if enabled:
        Path("debug").mkdir(exist_ok=True)
        df.to_csv(Path("debug") / filename)


def initialize_shear_walls_dataframe(
    shear_walls_dataframe: pd.DataFrame,
    seismic_loads: SeismicLoads,
    tributary_area_check: dict[int, float] | FloatLike,
) -> pd.DataFrame:
    """BE SURE TO NAME YOUR WALLS WITH '--' BETWEEN THE GRIDLINES SO THAT THE WALL LINE IS PROPERLY SELECTED.

    Args:
        shear_walls_dataframe (pd.DataFrame): _description_
        seismic_loads (SeismicLoads): _description_
        tributary_area_check (dict[int, float] | FloatLike): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = shear_walls_dataframe.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.lower()

    df = pd.concat(
        [df.assign(level=level) for level in seismic_loads.structure.shear_wall_levels[::-1]],
        ignore_index=True,
    )

    # Ensure a line column exists and populate it
    derived_line = df["wall"].str.split("--", n=1).str[0].str.strip()
    if "line" not in df.columns:
        df["line"] = derived_line
    else:
        mask = df["line"].isna() | (df["line"].astype(str).str.strip() == "")
        df.loc[mask, "line"] = derived_line[mask]

    # Fill empty sheathed sides with 1
    if "sheathed sides" not in df.columns:
        df["sheathed sides"] = 1
    else:
        df["sheathed sides"] = df["sheathed sides"].fillna(1)

    # Fill empty accidental torsional amplification factor with 1
    if "accidental torsional amplification factor x" not in df.columns:
        df["accidental torsional amplification factor x"] = 1
    else:
        df["accidental torsional amplification factor x"] = df["accidental torsional amplification factor x"].fillna(1)
    if "accidental torsional amplification factor y" not in df.columns:
        df["accidental torsional amplification factor y"] = 1
    else:
        df["accidental torsional amplification factor y"] = df["accidental torsional amplification factor y"].fillna(1)

    # # Fill empties with 0
    # shear_walls = shear_walls.fillna(0)

    # Index columns set
    df = df.rename(columns={"level": "Level", "wall": "Wall", "direction": "Direction"})
    df = df.set_index(["Level", "Wall"])

    # Add trib area automatically. Drop the trib width and trib length columns after area is determined
    if "tributary area" not in df.columns:
        df["tributary area"] = 0.0

    has_width = "tributary width" in df.columns
    has_length = "tributary length" in df.columns

    if has_width and has_length:
        missing_area = df["tributary area"] == 0

        df.loc[missing_area, "tributary area"] = (
            df.loc[missing_area, "tributary width"] * df.loc[missing_area, "tributary length"]
        )

        df.drop(
            columns=["tributary width", "tributary length"],
            inplace=True,
        )

    df["level tributary area"] = df.groupby(["Level", "Direction"])["tributary area"].transform("sum")

    # Automatically extract the shear line by splitting at "_" and taking the first section
    df["line"] = df.index.get_level_values("Wall").str.split("--").str[0]

    # Get tributary area and wall length for the shear line
    df["line tributary area"] = df.groupby(["Level", "Direction", "line"])["tributary area"].transform("sum")
    df["line wall length"] = df.groupby(["Level", "Direction", "line"])["wall length"].transform("sum")

    if "angle" not in df.columns:
        df["angle"] = pd.NA

    df.loc[df["angle"].isna(), "angle"] = df.loc[df["angle"].isna(), "Direction"].map({"x": 0, "y": 90})

    seismic_force_lookup = {
        "x": seismic_loads.seismic_loads_x["lateral seismic force"],
        "y": seismic_loads.seismic_loads_y["lateral seismic force"],
    }

    df["level seismic force"] = df.apply(
        lambda row: seismic_force_lookup[row["Direction"]].loc[row.name[0] + 1] * KIPS_TO_LBS, axis=1
    )

    diaphragm_force_lookup = {
        "x": seismic_loads.seismic_loads_x["diaphragm design force"],
        "y": seismic_loads.seismic_loads_y["diaphragm design force"],
    }

    df["level diaphragm force"] = df.apply(
        lambda row: diaphragm_force_lookup[row["Direction"]].loc[row.name[0] + 1] * KIPS_TO_LBS, axis=1
    )
    df["diaphragm force scale factor"] = df["level diaphragm force"] / df["level seismic force"]

    angle_rad = np.radians(df["angle"].astype(float))
    angle_factor = np.maximum(
        np.abs(np.cos(angle_rad)) + 0.3 * np.abs(np.sin(angle_rad)),
        np.abs(np.sin(angle_rad)) + 0.3 * np.abs(np.cos(angle_rad)),
    )

    df["level seismic force"] *= angle_factor

    level_forces = df.groupby(["Level", "Direction"], sort=False)["level seismic force"].first()

    cumulative_shear = level_forces.groupby("Direction").cumsum()

    df["total level cumulative shear demand"] = (
        df.index
        .to_frame(index=False)
        .assign(Direction=df["Direction"].to_numpy())
        .set_index(["Level", "Direction"])
        .index.map(cumulative_shear)
    )

    if tributary_area_check:
        _ = check_shear_wall_tributary_area(shear_walls_dataframe=df, tributary_area_check=tributary_area_check)
        df["level seismic force per area"] = df["level seismic force"] / df["level tributary area"]

    df["wall height"] = df.index.get_level_values("Level").map(seismic_loads.structure.levels_data["level height"])
    df["aspect ratio"] = df["wall height"] / df["wall length"]

    return df


def create_shear_walls_dataframe_from_dict(
    shear_wall_input_dict: dict,
    seismic_loads: SeismicLoads,
    tributary_area_check: dict[int, float] | float | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(shear_wall_input_dict, orient="index").rename_axis("wall").reset_index()
    df = initialize_shear_walls_dataframe(df, seismic_loads, tributary_area_check)
    return df


def create_shear_walls_dataframe_from_csv(
    shear_wall_input_csv: Path | str,
    seismic_loads: SeismicLoads,
    tributary_area_check: dict[int, float] | float | None = None,
) -> pd.DataFrame:
    """Generate the shear wall dataframe with the following assumptions:

    - Wood shear walls
    - All stacking shear walls

    If walls don't stack, you can edit the dataframe after it is created.

    Args:
        shear_wall_input_csv (Path | str): CSV with shear wall info
        seismic_loads (SeismicLoads): SeismicLoads object created from the equivalent lateral force procedure seismic loading creation (ELF)
        tributary_area_check (list[float] | float | None): list or single float for checking that all trib areas add up to the total for a level

    Returns:
        pd.DataFrame: Shear wall dataframe
    """
    df = pd.read_csv(shear_wall_input_csv)
    df = initialize_shear_walls_dataframe(df, seismic_loads, tributary_area_check)
    return df


def check_shear_wall_tributary_area(
    shear_walls_dataframe: pd.DataFrame,
    tributary_area_check: dict[tuple[int, str], float] | FloatLike,
) -> pd.Series:
    """Check total tributary area per level and direction against expected values.

    Args:
        shear_walls_dataframe: Shear wall dataframe with a "Level" index level, a
            "Direction" column, and a "tributary area" column.
        tributary_area_check: Expected tributary areas. A single value is applied
            to every (level, direction) group, or a dict keyed by (level,
            direction) tuples provides per-group expected values.

    Returns:
        Series of total tributary area indexed by (Level, Direction).

    Raises:
        ValueError: If check values are missing groups, contain extra groups, or
            any group's total tributary area does not match its expected value.
    """
    total_trib_area = shear_walls_dataframe.groupby(["Level", "Direction"])["tributary area"].sum()

    # Convert a single value into a dict for all (level, direction) groups
    if isinstance(tributary_area_check, FloatLike):
        tributary_area_check = {index: float(tributary_area_check) for index in total_trib_area.index}

    # Check that every (level, direction) group has a check value
    missing_groups = set(total_trib_area.index) - set(tributary_area_check)
    extra_groups = set(tributary_area_check) - set(total_trib_area.index)

    if missing_groups:
        raise ValueError(f"Trib area check missing groups: {sorted(missing_groups)}")

    if extra_groups:
        raise ValueError(f"Trib area check contains extra groups: {sorted(extra_groups)}")

    failures: list[str] = []

    for (level, direction), actual_area in total_trib_area.items():
        expected_area = tributary_area_check[(level, direction)]

        if not math.isclose(
            actual_area,
            expected_area,
            rel_tol=1e-3,
            abs_tol=1e-1,
        ):
            failures.append(
                f"Level {level} {direction.upper()}: "
                f"expected {expected_area:.3f}, got {actual_area:.3f}\n"
                "Enter manual tributary areas (or updated tributary widths and "
                "tributary lengths) in the input CSV file to match the correct total area."
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
    df = shear_walls_dataframe
    for wall, properties in mapping_dict.items():
        for column, value in properties.items():
            df.loc[
                df.index.get_level_values("Wall") == wall,
                column,
            ] = value
    return df


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
    df = shear_walls_dataframe
    for level, properties in mapping_dict.items():
        for column, value in properties.items():
            df.loc[
                df.index.get_level_values("Level") == level,
                column,
            ] = value
    return df


def _calculate_flexible_seismic_demand(shear_walls_dataframe: pd.DataFrame):
    df = shear_walls_dataframe
    necessary_columns = ["line", "tributary area", "wall length", "level seismic force per area"]
    _check_for_missing_columns(df, necessary_columns)
    df["line shear demand"] = df["line tributary area"] * df["level seismic force per area"]
    df["line unit shear"] = df["line shear demand"] / df["line wall length"]
    df["line cumulative shear demand"] = df.groupby(level="Wall")["line shear demand"].cumsum()
    df["line cumulative unit shear"] = df["line cumulative shear demand"] / df["line wall length"]

    df["shear demand"] = df["line unit shear"] * df["wall length"]
    df["cumulative shear demand"] = df.groupby(level="Wall")["shear demand"].cumsum()
    df["flexible shear force demand"] = df["cumulative shear demand"]
    df["unit shear demand"] = df["cumulative shear demand"] / df["wall length"]
    df["unit flexible shear demand"] = df["unit shear demand"]
    df["adjusted unit shear demand"] = df["unit shear demand"] * ASCE_ASD_8_E_H
    df["adjusted flexible unit shear demand"] = df["adjusted unit shear demand"]

    return df


def _choose_shear_wall_sheathing(
    shear_walls_dataframe: pd.DataFrame,
    sheathing: Sheathing,
    dcr_check_value: float = 1.0,
    default_one_sided_sheathing: bool = True,
) -> pd.DataFrame:
    df = shear_walls_dataframe

    necessary_columns = ["adjusted unit shear demand"]
    _check_for_missing_columns(df, necessary_columns)

    result_columns = [
        "adjusted unit shear capacity",
        "nail spacing",
        "sheathing shear stiffness",
        "shear dcr",
    ]

    if default_one_sided_sheathing:
        df["sheathed sides"] = 1

    df[result_columns] = get_sheathing_properties(
        df[["adjusted unit shear demand", "sheathed sides"]],
        sheathing=sheathing,
    )

    requires_two_sides = df["shear dcr"] > dcr_check_value

    if requires_two_sides.any():
        df.loc[requires_two_sides, "sheathed sides"] = 2

        df.loc[requires_two_sides, result_columns] = get_sheathing_properties(
            df.loc[
                requires_two_sides,
                ["adjusted unit shear demand", "sheathed sides"],
            ],
            sheathing=sheathing,
        )

    return df


def design_shear_walls_flexible_assumption(
    shear_walls_dataframe: pd.DataFrame, sheathing: Sheathing, dcr_check_value: float = 1
):
    shear_walls_dataframe = _calculate_flexible_seismic_demand(shear_walls_dataframe)
    return _choose_shear_wall_sheathing(shear_walls_dataframe, sheathing, dcr_check_value)


def _check_for_missing_columns(dataframe: pd.DataFrame, necessary_columns: list[str]) -> None:
    missing_columns = [column for column in necessary_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"The supplied dataframe is missing the following columns:\n{missing_columns}")


def _calculate_shear_wall_stiffness(
    shear_walls_dataframe: pd.DataFrame,
    end_post_youngs_modulus: float | None,
    end_post_area: float | None,
    Delta_A: float | None,
    deflection_unit_conversion_factor: float = 1 / 12,
    stiffness_method: Literal["deflection", "length"] = "deflection",
):
    df = shear_walls_dataframe

    if stiffness_method == "deflection":
        necessary_columns = [
            "unit shear demand",
            "wall length",
            "wall height",
            "sheathing shear stiffness",
            "sheathed sides",
            "nail spacing",
            "cumulative shear demand",
        ]
        _check_for_missing_columns(df, necessary_columns)

        df["end post youngs modulus"] = end_post_youngs_modulus
        df["end post area"] = end_post_area
        df["Delta_A"] = Delta_A

        df["delta_sw"] = (
            8
            * df["unit shear demand"]
            * df["wall height"] ** 3
            / (df["end post youngs modulus"] * df["end post area"] * df["wall length"])
            + df["unit shear demand"]
            * df["wall height"]
            / (1000 * df["sheathing shear stiffness"] * df["sheathed sides"])
            + df["wall height"] * df["Delta_A"] / df["wall length"]
        ) * deflection_unit_conversion_factor

        df["wall stiffness"] = df["cumulative shear demand"] / df["delta_sw"]
        df["total level cumulative shear demand"] = df.groupby(["Level", "Direction"])[
            "cumulative shear demand"
        ].transform("sum")

    elif stiffness_method == "length":
        necessary_columns = [
            "wall length",
        ]
        _check_for_missing_columns(df, necessary_columns)
        # Relative stiffness proportional to wall length.
        # Absolute magnitude is irrelevant because only the
        # relative stiffness is used for force distribution.
        df["delta_sw"] = np.nan
        df["wall stiffness"] = df["wall length"]

    else:
        raise ValueError("stiffness_method must be 'deflection' or 'length'")

    df["total level stiffness"] = df.groupby(["Level", "Direction"])["wall stiffness"].transform("sum")

    df["relative wall stiffness"] = df["wall stiffness"] / df["total level stiffness"]

    return df


def _calculate_rigid_direct_shear_demand(shear_walls_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = shear_walls_dataframe
    necessary_columns = ["relative wall stiffness", "total level cumulative shear demand"]
    _check_for_missing_columns(df, necessary_columns)

    df["direct shear force"] = df["total level cumulative shear demand"] * df["relative wall stiffness"]
    return df


def _calculate_center_of_rigidity(shear_walls_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = shear_walls_dataframe
    necessary_columns = ["relative wall stiffness", "x", "y"]
    _check_for_missing_columns(df, necessary_columns)

    df["kx"] = df["relative wall stiffness"] * df["x"]
    df["ky"] = df["relative wall stiffness"] * df["y"]
    # all relative wall rigidities must sum to 1
    df["CRx"] = df.groupby(["Level", "Direction"])["kx"].transform("sum")
    df["CRy"] = df.groupby(["Level", "Direction"])["ky"].transform("sum")

    # distance from CR
    df["dx"] = abs(df["x"] - df["CRx"])
    df["dy"] = abs(df["y"] - df["CRy"])

    return df


def _calculate_eccentricities(
    shear_walls_dataframe: pd.DataFrame, center_of_mass: tuple[float, float], plan_dimensions: tuple[float, float]
) -> pd.DataFrame:
    df = shear_walls_dataframe
    necessary_columns = [
        "CRx",
        "CRy",
        "accidental torsional amplification factor x",
        "accidental torsional amplification factor y",
    ]
    _check_for_missing_columns(df, necessary_columns)

    plan_dim_x = plan_dimensions[0]
    plan_dim_y = plan_dimensions[1]
    CMx = center_of_mass[0]
    CMy = center_of_mass[1]

    # x eccentricity. Use the "+" eccentricity for walls on the + side of the CR. "-" for - side
    accidental_multiplier_x = 0.05 * df["accidental torsional amplification factor x"]
    CMxa_plus = CMx + plan_dim_x * accidental_multiplier_x
    CMxa_minus = CMx - plan_dim_x * accidental_multiplier_x
    df["ex"] = np.where(
        df["x"] <= df["CRx"],
        df["CRx"] - CMxa_minus,
        CMxa_plus - df["CRx"],
    )
    df["ex"] = np.where(
        df["ex"] <= 0, plan_dim_x * accidental_multiplier_x, df["ex"]
    )  # assign 5% plan dim to negative values

    # y eccentricity. Use the "+" eccentricity for walls on the + side of the CR. "-" for - side
    accidental_multiplier_y = 0.05 * df["accidental torsional amplification factor y"]
    CMya_plus = CMy + plan_dim_y * accidental_multiplier_y
    CMya_minus = CMy - plan_dim_y * accidental_multiplier_y
    df["ey"] = np.where(
        df["y"] <= df["CRy"],
        df["CRy"] - CMya_minus,
        CMya_plus - df["CRy"],
    )
    df["ey"] = np.where(
        df["ey"] <= 0, plan_dim_y * accidental_multiplier_y, df["ey"]
    )  # assign 5% plan dim to negative values

    return df


def _calculate_rigid_shear_demand(shear_walls_dataframe: pd.DataFrame):
    df = shear_walls_dataframe
    necessary_columns = [
        "total level cumulative shear demand",
        "ex",
        "ey",
        "dx",
        "dy",
        "wall stiffness",
    ]
    _check_for_missing_columns(df, necessary_columns)

    # Torque
    df["torque x"] = df["total level cumulative shear demand"] * df["ey"]
    df["torque y"] = df["total level cumulative shear demand"] * df["ex"]

    # Polar moment of inertia
    df["Jx wall"] = df["wall stiffness"] * df["dx"] ** 2
    df["Jy wall"] = df["wall stiffness"] * df["dy"] ** 2
    df["J"] = (df["Jx wall"] + df["Jy wall"]).groupby(["Level"]).transform("sum")

    df["torsional shear force x"] = df["torque x"] * df["wall stiffness"] * df["dx"] / df["J"]
    df["torsional shear force y"] = df["torque y"] * df["wall stiffness"] * df["dy"] / df["J"]
    df["torsional shear force"] = np.where(
        df["torsional shear force x"] == 0,
        df["torsional shear force y"],
        df["torsional shear force x"],
    )
    df = _calculate_rigid_direct_shear_demand(df)
    df["rigid shear force demand"] = df["torsional shear force"] + df["direct shear force"]

    return df


def _envelope_shear_demand(shear_walls_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = shear_walls_dataframe
    # necessary_columns = ["flexible shear force demand", "rigid shear force demand"]
    # _check_for_missing_columns(df, necessary_columns)

    if "flexible shear force demand" not in df.columns:
        df["shear force demand"] = df["rigid shear force demand"]
    elif "rigid shear force demand" not in df.columns:
        df["shear force demand"] = df["flexible shear force demand"]
    else:
        df["shear force demand"] = np.where(
            df["flexible shear force demand"] >= df["rigid shear force demand"],
            df["flexible shear force demand"],
            df["rigid shear force demand"],
        )
    df["unit shear demand"] = df["shear force demand"] / df["wall length"]
    df["floor shear demand"] = abs(df.groupby("Wall")["shear force demand"].diff().fillna(df["shear force demand"]))
    df["floor unit shear demand"] = df["floor shear demand"] / df["wall length"]
    df["diaphragm floor unit shear demand"] = df["diaphragm force scale factor"] * df["floor unit shear demand"]
    df["adjusted diaphragm floor unit shear demand"] = df["diaphragm floor unit shear demand"] * ASCE_ASD_8_E_H
    df["adjusted unit shear demand"] = df["shear force demand"] / df["wall length"] * ASCE_ASD_8_E_H
    df["adjusted floor unit shear demand"] = df["floor unit shear demand"] * ASCE_ASD_8_E_H

    return df


def _check_shear_dcr(
    shear_walls_dataframe: pd.DataFrame,
    max_dcr: float = 1.0,
) -> tuple[bool, float, pd.DataFrame]:
    """Check shear wall DCRs and return failures.

    Args:
        shear_walls_dataframe: Shear wall dataframe.
        max_dcr: Maximum acceptable DCR.

    Returns:
        Tuple containing:
            - Whether all walls pass.
            - Maximum DCR.
            - DataFrame of walls exceeding max_dcr.
    """
    df = shear_walls_dataframe

    necessary_columns = ["shear dcr"]
    _check_for_missing_columns(df, necessary_columns)

    failing_walls = df.loc[df["shear dcr"] > max_dcr].copy()

    return (
        failing_walls.empty,
        df["shear dcr"].max(),
        failing_walls,
    )


def assign_shear_wall_type(df: pd.DataFrame) -> pd.DataFrame:
    """Assign shear wall type from sheathed sides and nail spacing."""

    wall_type_lookup = {
        (1, 6): 1,
        (1, 4): 2,
        (1, 3): 3,
        (1, 2): 4,
        (2, 4): 5,
        (2, 3): 6,
        (2, 2): 7,
    }

    df["shear wall type"] = pd.MultiIndex.from_arrays([
        df["sheathed sides"].astype(int),
        df["nail spacing"].astype(int),
    ]).map(wall_type_lookup)

    if df["shear wall type"].isna().any():
        invalid_rows = df.loc[
            df["shear wall type"].isna(),
            ["sheathed sides", "nail spacing"],
        ].drop_duplicates()

        raise ValueError(f"Invalid shear wall type combination(s):\n{invalid_rows.to_string(index=False)}")

    df["shear wall type"] = df["shear wall type"].astype(int)

    return df


def design_shear_walls_envelope(
    shear_walls_dataframe: pd.DataFrame,
    sheathing: Sheathing,
    end_post_youngs_modulus: float,
    end_post_area: float,
    Delta_A: float,
    center_of_mass: tuple[float, float],
    plan_dimensions: tuple[float, float],
    c_d_x: float,
    c_d_y: float,
    i_e: float,
    dcr_check_value: float = 1,
    allowable_story_drift_coefficient: float = 0.020,
    stiffness_method: Literal["deflection", "length"] = "deflection",
    debug_save_intermediate_csvs: bool = False,
):
    df = shear_walls_dataframe
    df = design_shear_walls_flexible_assumption(df, sheathing, dcr_check_value)
    df["adjusted flexible unit shear capacity"] = df["adjusted unit shear capacity"]
    df["sheathed sides flexible"] = df["sheathed sides"]
    df["nail spacing flexible"] = df["nail spacing"]
    _debug_save(df, "01_flexible_design.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_shear_wall_stiffness(
        df, end_post_youngs_modulus, end_post_area, Delta_A, stiffness_method=stiffness_method
    )
    _debug_save(df, "02_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_center_of_rigidity(df)
    _debug_save(df, "03_calculate_center_of_rigidity.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_eccentricities(df, center_of_mass, plan_dimensions)
    _debug_save(df, "04_calculate_eccentricities.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_rigid_shear_demand(df)
    _debug_save(df, "05_calculate_rigid_shear_demand.csv", enabled=debug_save_intermediate_csvs)
    df = _envelope_shear_demand(df)
    _debug_save(df, "06_envelope_shear_demand.csv", enabled=debug_save_intermediate_csvs)
    df = _choose_shear_wall_sheathing(df, sheathing, dcr_check_value)
    _debug_save(df, "07_choose_shear_wall_sheathing.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_shear_wall_stiffness(
        df, end_post_youngs_modulus, end_post_area, Delta_A, stiffness_method=stiffness_method
    )  # update the stiffnesses with updated walls
    _debug_save(df, "08_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)
    df = calculate_deflections(
        shear_walls_dataframe,
        plan_dimensions,
        c_d_x,
        c_d_y,
        i_e,
        allowable_story_drift_coefficient,
    )
    _debug_save(df, "08_calculate_deflections.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_eccentricities(df, center_of_mass, plan_dimensions)
    _debug_save(df, "10_calculate_eccentricities.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_rigid_shear_demand(df)
    _debug_save(df, "11_calculate_rigid_shear_demand.csv", enabled=debug_save_intermediate_csvs)
    df = _envelope_shear_demand(df)
    _debug_save(df, "12_envelope_shear_demand.csv", enabled=debug_save_intermediate_csvs)
    df = _choose_shear_wall_sheathing(df, sheathing, dcr_check_value)
    _debug_save(df, "13_choose_shear_wall_sheathing.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_shear_wall_stiffness(
        df, end_post_youngs_modulus, end_post_area, Delta_A, stiffness_method=stiffness_method
    )  # update the stiffnesses with updated walls
    _debug_save(df, "14_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)

    # Find walls that ended up with a nonsensical configuration:
    # 2-sided sheathing with 6" edge nailing.
    mask = (df["sheathed sides"] == 2) & (df["nail spacing"] == 6)
    if mask.any():
        # Force these walls back to a one-sided schedule.
        df.loc[mask, "sheathed sides"] = 1

        # Re-run the stiffness / force distribution / wall selection cycle.
        df = _calculate_shear_wall_stiffness(
            # Re-run the stiffness / force distribution / wall selection cycle.
            df,
            end_post_youngs_modulus,
            end_post_area,
            Delta_A,
            stiffness_method=stiffness_method,
        )
        _debug_save(df, "15_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)
        df = _calculate_center_of_rigidity(df)
        _debug_save(df, "16_calculate_center_of_rigidity.csv", enabled=debug_save_intermediate_csvs)
        df = _calculate_eccentricities(df, center_of_mass, plan_dimensions)
        _debug_save(df, "17_calculate_eccentricities.csv", enabled=debug_save_intermediate_csvs)
        df = _calculate_rigid_shear_demand(df)
        _debug_save(df, "18_calculate_rigid_shear_demand.csv", enabled=debug_save_intermediate_csvs)
        df = _envelope_shear_demand(df)
        _debug_save(df, "19_envelope_shear_demand.csv", enabled=debug_save_intermediate_csvs)
        df = _choose_shear_wall_sheathing(df, sheathing, dcr_check_value)
        _debug_save(df, "20_choose_shear_wall_sheathing.csv", enabled=debug_save_intermediate_csvs)
        df = _calculate_shear_wall_stiffness(
            df, end_post_youngs_modulus, end_post_area, Delta_A, stiffness_method=stiffness_method
        )
        _debug_save(df, "21_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)
        df = calculate_deflections(df, plan_dimensions, c_d_x, c_d_y, i_e, allowable_story_drift_coefficient)
        _debug_save(df, "22_calculate_deflections.csv", enabled=debug_save_intermediate_csvs)

    dcr_check, max_shear_dcr, failing_walls = _check_shear_dcr(df)
    if not dcr_check:
        raise ValueError(
            f"The following walls are failing: \n{failing_walls['shear dcr']}\nYour highest shear DCR is {max_shear_dcr} > 1, please refine your design to add more shear wall or remove the bad wall."
        )

    assign_shear_wall_type(df)
    _debug_save(df, "23_assign_shear_wall_type.csv", enabled=debug_save_intermediate_csvs)

    return df


def design_shear_walls_rigid(
    shear_walls_dataframe: pd.DataFrame,
    sheathing: Sheathing,
    center_of_mass: tuple[float, float],
    plan_dimensions: tuple[float, float],
    c_d_x: float,
    c_d_y: float,
    i_e: float,
    dcr_check_value: float = 1,
    allowable_story_drift_coefficient: float = 0.020,
    # stiffness_method: Literal["deflection", "length"] = "length",
    debug_save_intermediate_csvs: bool = False,
):
    df = shear_walls_dataframe
    # df = design_shear_walls_flexible_assumption(df, sheathing, dcr_check_value)
    # df["adjusted flexible unit shear capacity"] = df["adjusted unit shear capacity"]
    # df["sheathed sides flexible"] = df["sheathed sides"]
    # df["nail spacing flexible"] = df["nail spacing"]
    # _debug_save(df, "01_flexible_design.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_shear_wall_stiffness(
        df,
        end_post_youngs_modulus=None,
        end_post_area=None,
        Delta_A=None,
        stiffness_method="length",
    )
    _debug_save(df, "01_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_center_of_rigidity(df)
    _debug_save(df, "02_calculate_center_of_rigidity.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_eccentricities(df, center_of_mass, plan_dimensions)
    _debug_save(df, "03_calculate_eccentricities.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_rigid_shear_demand(df)
    _debug_save(df, "04_calculate_rigid_shear_demand.csv", enabled=debug_save_intermediate_csvs)
    df = _envelope_shear_demand(df)
    _debug_save(df, "05_envelope_shear_demand.csv", enabled=debug_save_intermediate_csvs)
    df = _choose_shear_wall_sheathing(df, sheathing, dcr_check_value)
    _debug_save(df, "06_choose_shear_wall_sheathing.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_shear_wall_stiffness(
        df, end_post_youngs_modulus=None, end_post_area=None, Delta_A=None, stiffness_method="length"
    )  # update the stiffnesses with updated walls
    _debug_save(df, "07_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)
    df = calculate_deflections(
        shear_walls_dataframe,
        plan_dimensions,
        c_d_x,
        c_d_y,
        i_e,
        allowable_story_drift_coefficient,
    )
    _debug_save(df, "08_calculate_deflections.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_eccentricities(df, center_of_mass, plan_dimensions)
    _debug_save(df, "09_calculate_eccentricities.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_rigid_shear_demand(df)
    _debug_save(df, "10_calculate_rigid_shear_demand.csv", enabled=debug_save_intermediate_csvs)
    df = _envelope_shear_demand(df)
    _debug_save(df, "11_envelope_shear_demand.csv", enabled=debug_save_intermediate_csvs)
    df = _choose_shear_wall_sheathing(df, sheathing, dcr_check_value)
    _debug_save(df, "12_choose_shear_wall_sheathing.csv", enabled=debug_save_intermediate_csvs)
    df = _calculate_shear_wall_stiffness(
        df, end_post_youngs_modulus=None, end_post_area=None, Delta_A=None, stiffness_method="length"
    )  # update the stiffnesses with updated walls
    _debug_save(df, "13_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)

    # Find walls that ended up with a nonsensical configuration:
    # 2-sided sheathing with 6" edge nailing.
    mask = (df["sheathed sides"] == 2) & (df["nail spacing"] == 6)
    if mask.any():
        # Force these walls back to a one-sided schedule.
        df.loc[mask, "sheathed sides"] = 1

        # Re-run the stiffness / force distribution / wall selection cycle.
        df = _calculate_shear_wall_stiffness(
            # Re-run the stiffness / force distribution / wall selection cycle.
            df,
            end_post_youngs_modulus=None,
            end_post_area=None,
            Delta_A=None,
            stiffness_method="length",
        )
        _debug_save(df, "14_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)
        df = _calculate_center_of_rigidity(df)
        _debug_save(df, "15_calculate_center_of_rigidity.csv", enabled=debug_save_intermediate_csvs)
        df = _calculate_eccentricities(df, center_of_mass, plan_dimensions)
        _debug_save(df, "16_calculate_eccentricities.csv", enabled=debug_save_intermediate_csvs)
        df = _calculate_rigid_shear_demand(df)
        _debug_save(df, "17_calculate_rigid_shear_demand.csv", enabled=debug_save_intermediate_csvs)
        df = _envelope_shear_demand(df)
        _debug_save(df, "18_envelope_shear_demand.csv", enabled=debug_save_intermediate_csvs)
        df = _choose_shear_wall_sheathing(df, sheathing, dcr_check_value)
        _debug_save(df, "19_choose_shear_wall_sheathing.csv", enabled=debug_save_intermediate_csvs)
        df = _calculate_shear_wall_stiffness(
            df, end_post_youngs_modulus=None, end_post_area=None, Delta_A=None, stiffness_method="length"
        )
        _debug_save(df, "20_calculate_shear_wall_stiffness.csv", enabled=debug_save_intermediate_csvs)
        df = calculate_deflections(df, plan_dimensions, c_d_x, c_d_y, i_e, allowable_story_drift_coefficient)
        _debug_save(df, "21_calculate_deflections.csv", enabled=debug_save_intermediate_csvs)

    dcr_check, max_shear_dcr, failing_walls = _check_shear_dcr(df)
    if not dcr_check:
        raise ValueError(
            f"The following walls are failing: \n{failing_walls['shear dcr']}\nYour highest shear DCR is {max_shear_dcr} > 1, please refine your design to add more shear wall or remove the bad wall."
        )

    assign_shear_wall_type(df)
    _debug_save(df, "22_assign_shear_wall_type.csv", enabled=debug_save_intermediate_csvs)

    return df


def _calculate_end_post_tension_forces(
    shear_walls_dataframe: pd.DataFrame,
    dead_load_per_area: float,
    s_ds: float,
    wall_arm_shortening_length: float = 0.5,
    tributary_area_reduction_factor_dict: dict[str, dict[str, float]] | None = None,
    rho: float = 1.0,
) -> pd.DataFrame:
    df = shear_walls_dataframe
    necessary_columns = ["tributary area", "wall height", "wall length", "shear force demand"]
    _check_for_missing_columns(df, necessary_columns)

    df["gravity tributary area"] = df["tributary area"]
    if tributary_area_reduction_factor_dict:
        wall_factors = df.index.get_level_values("Wall").map(tributary_area_reduction_factor_dict).fillna(1.0)

        df["gravity tributary area"] *= wall_factors

    df["tension wall vertical load length"] = np.minimum(
        df["wall height"],  # take only height of wall for long walls (aspect<=1:1)
        df["wall length"],  # use full wall length for short walls (aspect>1:1)
    )

    df["dead load"] = (
        (df["gravity tributary area"] * dead_load_per_area / df["wall length"])  # dead load per length
        * df["tension wall vertical load length"]
    )

    df["horizontal effects moment lc10"] = rho * ASCE_ASD_10_E_H * df["shear force demand"] * df["wall height"]
    df["connors floor seismic axial load"] = rho * df["shear force demand"] * df["wall height"] / df["wall length"]
    df["connors seismic axial load"] = df.groupby(["Wall"])["connors floor seismic axial load"].cumsum()

    df["vertical effects moment lc10"] = (
        ASCE_ASD_10_E_V * 0.2 * s_ds * df["dead load"] * (df["tension wall vertical load length"] / 2)
    )

    df["dead effects moment lc10"] = ASCE_ASD_10_D * df["dead load"] * (df["tension wall vertical load length"] / 2)

    df["net moment lc10"] = (
        df["horizontal effects moment lc10"] + df["vertical effects moment lc10"] - df["dead effects moment lc10"]
    )
    df["floor tension force lc10"] = df["net moment lc10"] / (df["wall length"] - wall_arm_shortening_length)

    df["tension force lc10"] = df.groupby(["Wall"])["floor tension force lc10"].cumsum()
    df["tension force"] = df["tension force lc10"]

    return df


def _calculate_end_post_compression_forces(
    shear_walls_dataframe: pd.DataFrame,
    dead_load_per_area: float,
    s_ds: float,
    wall_arm_shortening_length: float = 0.5,
    tributary_area_reduction_factor_dict: dict[str, dict[str, float]] | None = None,
    stud_bay_width: float = 16 / 12,
    live_load_per_area: float = 40,
    snow_load_per_area: float = 25,
    rho: float = 1.0,
) -> pd.DataFrame:
    df = shear_walls_dataframe
    necessary_columns = ["tributary area", "wall height", "wall length", "shear force demand"]
    _check_for_missing_columns(df, necessary_columns)

    df["gravity tributary area"] = df["tributary area"]
    if tributary_area_reduction_factor_dict:
        wall_factors = df.index.get_level_values("Wall").map(tributary_area_reduction_factor_dict).fillna(1.0)

        df["gravity tributary area"] *= wall_factors

    df["compression wall vertical load length"] = stud_bay_width

    df["dead load"] = (
        (df["gravity tributary area"] * dead_load_per_area / df["wall length"])  # dead load per length
        * df["compression wall vertical load length"]
    )

    df["live load"] = (
        (df["gravity tributary area"] * live_load_per_area / df["wall length"])  # dead load per length
        * df["compression wall vertical load length"]
    )

    df["snow load"] = (
        (df["gravity tributary area"] * snow_load_per_area / df["wall length"])  # dead load per length
        * df["compression wall vertical load length"]
    )

    df["horizontal effects moment lc8"] = rho * ASCE_ASD_8_E_H * df["shear force demand"] * df["wall height"]
    df["vertical effects moment lc8"] = (
        ASCE_ASD_8_E_V * 0.2 * s_ds * df["dead load"] * (df["compression wall vertical load length"] / 2)
    )
    df["dead effects moment lc8"] = ASCE_ASD_8_D * df["dead load"] * (df["compression wall vertical load length"] / 2)

    df["horizontal effects moment lc9"] = rho * ASCE_ASD_9_E_H * df["shear force demand"] * df["wall height"]
    df["vertical effects moment lc9"] = (
        ASCE_ASD_9_E_V * 0.2 * s_ds * df["dead load"] * (df["compression wall vertical load length"] / 2)
    )

    df["dead effects moment lc9"] = ASCE_ASD_9_D * df["dead load"] * (df["compression wall vertical load length"] / 2)
    df["live effects moment lc9"] = ASCE_ASD_9_L * df["live load"] * (df["compression wall vertical load length"] / 2)
    df["snow effects moment lc9"] = ASCE_ASD_9_S * df["snow load"] * (df["compression wall vertical load length"] / 2)

    df["net moment lc8"] = (
        df["horizontal effects moment lc8"] + df["vertical effects moment lc8"] + df["dead effects moment lc8"]
    )

    df["net moment lc9"] = (
        df["horizontal effects moment lc9"]
        + df["vertical effects moment lc9"]
        + df["dead effects moment lc9"]
        + df["live effects moment lc9"]
        + df["snow effects moment lc9"]
    )

    df["floor compression force lc8"] = df["net moment lc8"] / (df["wall length"] - wall_arm_shortening_length)
    df["floor compression force lc9"] = df["net moment lc9"] / (df["wall length"] - wall_arm_shortening_length)

    df["compression force lc8"] = df.groupby(["Wall"])["floor compression force lc8"].cumsum()
    df["compression force lc9"] = df.groupby(["Wall"])["floor compression force lc9"].cumsum()

    df["compression force"] = np.maximum(df["compression force lc8"], df["compression force lc9"])

    return df


def calculate_end_post_forces(
    shear_walls_dataframe: pd.DataFrame,
    dead_load_per_area: float,
    s_ds: float,
    wall_arm_shortening_length: float = 0.5,
    tributary_area_reduction_factor_dict: dict[str, dict[str, float]] | None = None,
    stud_bay_width: float = 16 / 12,
    live_load_per_area: float = 40,
    snow_load_per_area: float = 25,
) -> pd.DataFrame:
    df = shear_walls_dataframe
    df = _calculate_end_post_tension_forces(
        df,
        dead_load_per_area,
        s_ds,
        wall_arm_shortening_length,
        tributary_area_reduction_factor_dict=tributary_area_reduction_factor_dict,
    )
    df = _calculate_end_post_compression_forces(
        df,
        dead_load_per_area,
        s_ds,
        wall_arm_shortening_length,
        tributary_area_reduction_factor_dict=tributary_area_reduction_factor_dict,
        stud_bay_width=stud_bay_width,
        live_load_per_area=live_load_per_area,
        snow_load_per_area=snow_load_per_area,
    )
    return df


def assign_force_schedule(
    shear_walls_dataframe: pd.DataFrame,
    num_options: int = 5,
    level_name: str = "Level",
    tension_column: str = "tension force",
    compression_column: str = "compression force",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign force schedules by grouping forces into bins.

    Args:
        shear_walls_dataframe: Input dataframe.
        num_options: Maximum number of schedule options (1-26).
        level_name: Index level used to group rows.
        tension_column: Column containing tension forces (lbs).
        compression_column: Column containing compression forces (lbs).

    Returns:
        Tuple containing:
            - Updated dataframe with assigned schedule options.
            - Schedule dataframe indexed by level and option.
    """
    if not 1 <= num_options <= 26:
        raise ValueError("num_options must be between 1 and 26.")

    labels = list(ascii_uppercase[:num_options])
    schedule_rows = []

    def _round_schedule_force(force: pd.Series) -> pd.Series:
        """Round schedule forces.

        < 10 kip  -> round up to next 0.5 kip
        >= 10 kip -> round up to next whole kip
        """
        rounded = np.ceil(force * 2) / 2

        mask = rounded >= 10
        rounded[mask] = np.ceil(rounded[mask])

        return rounded

    def _assign_schedule(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()

        level = group.index.get_level_values(level_name)[0]
        n = len(group)

        # Assign bins
        group["_bin"] = np.floor(np.arange(n) * min(num_options, n) / n).astype(int)

        # Get maximum forces in each bin
        schedule = (
            group
            .groupby("_bin")
            .agg(
                tension=(tension_column, "max"),
                compression=(compression_column, "max"),
            )
            .reset_index()
        )

        # Convert to kips
        schedule["tension"] /= 1000.0
        schedule["compression"] /= 1000.0

        # Sort by governing force
        schedule["governing"] = schedule[["tension", "compression"]].max(axis=1)

        schedule = schedule.sort_values("governing").reset_index(drop=True)

        # Enforce monotonic schedules
        schedule["tension"] = schedule["tension"].cummax()
        schedule["compression"] = schedule["compression"].cummax()

        # Round schedule values
        schedule["tension"] = _round_schedule_force(schedule["tension"])
        schedule["compression"] = _round_schedule_force(schedule["compression"])

        schedule["Option"] = labels[: len(schedule)]
        schedule["Level"] = level

        # Map bins back to option letters
        bin_to_option = dict(
            zip(
                schedule["_bin"],
                schedule["Option"],
                strict=False,
            )
        )

        group["Option"] = group["_bin"].map(bin_to_option)

        schedule_map = schedule.set_index("Option")[["tension", "compression"]]

        group["binned tension force"] = group["Option"].map(schedule_map["tension"])

        group["binned compression force"] = group["Option"].map(schedule_map["compression"])

        schedule_rows.append(
            schedule[
                [
                    "Level",
                    "Option",
                    "tension",
                    "compression",
                ]
            ]
        )

        return group.drop(columns="_bin")

    shear_walls_dataframe = shear_walls_dataframe.groupby(
        level=level_name,
        group_keys=False,
    ).apply(_assign_schedule)

    schedule_df = (
        pd
        .concat(schedule_rows, ignore_index=True)
        .rename(
            columns={
                "tension": "binned tension force",
                "compression": "binned compression force",
            }
        )
        .set_index(["Level", "Option"])
        .sort_index()
    )

    return shear_walls_dataframe, schedule_df


def _calculate_deflection_at_center_of_mass(shear_walls_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = shear_walls_dataframe
    # no need to recalc shear demand from envelope as that would be overly conservative from the standpoint of the building
    df["level deflection at center of mass"] = (
        shear_walls_dataframe["total level cumulative shear demand"] / shear_walls_dataframe["total level stiffness"]
    )
    return df


def _calculate_rotation_at_center_of_rigidity(shear_walls_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = shear_walls_dataframe
    # no need to recalc shear demand from envelope as that would be overly conservative from the standpoint of the building
    df["level rotation at center of rigidity from torque x"] = df["torque x"] / df["J"]
    df["level rotation at center of rigidity from torque y"] = df["torque y"] / df["J"]
    df["level rotation at center of rigidity"] = np.maximum(
        df["level rotation at center of rigidity from torque x"],
        df["level rotation at center of rigidity from torque y"],
    )
    return df


def _calculate_deflection_at_diaphragm_edges_from_rotation(
    shear_walls_dataframe: pd.DataFrame, plan_dimensions: tuple[float, float]
) -> pd.DataFrame:
    df = shear_walls_dataframe
    df["x edge 1 to CRx"] = plan_dimensions[0] - df["CRx"]
    df["x edge 2 to CRx"] = df["CRx"] - plan_dimensions[0]
    # x deflections get a -1 factor. multiply it here into the y
    df["y edge 1 to CRy"] = plan_dimensions[1] - df["CRy"]
    df["y edge 2 to CRy"] = df["CRy"] - plan_dimensions[1]
    df["level edge 1 x deflection from ccw rotation"] = (
        df["level rotation at center of rigidity"] * df["y edge 1 to CRy"]
    )
    df["level edge 2 x deflection from ccw rotation"] = (
        df["level rotation at center of rigidity"] * df["y edge 2 to CRy"]
    )
    df["level edge 1 y deflection from ccw rotation"] = (
        df["level rotation at center of rigidity"] * df["x edge 1 to CRx"]
    )
    df["level edge 2 y deflection from ccw rotation"] = (
        df["level rotation at center of rigidity"] * df["x edge 2 to CRx"]
    )
    df["level edge 1 x deflection from cw rotation"] = (
        -1 * df["level rotation at center of rigidity"] * df["y edge 1 to CRy"]
    )
    df["level edge 2 x deflection from cw rotation"] = (
        -1 * df["level rotation at center of rigidity"] * df["y edge 2 to CRy"]
    )
    df["level edge 1 y deflection from cw rotation"] = (
        -1 * df["level rotation at center of rigidity"] * df["x edge 1 to CRx"]
    )
    df["level edge 2 y deflection from cw rotation"] = (
        -1 * df["level rotation at center of rigidity"] * df["x edge 2 to CRx"]
    )
    return df


def _calculate_total_deflection_at_diaphragm_edges(shear_walls_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = shear_walls_dataframe
    df["total edge 1 x deflection from ccw rotation"] = (
        df["level deflection at center of mass"] + df["level edge 1 x deflection from ccw rotation"]
    )
    df["total edge 2 x deflection from ccw rotation"] = (
        df["level deflection at center of mass"] + df["level edge 2 x deflection from ccw rotation"]
    )
    df["total edge 1 y deflection from ccw rotation"] = (
        df["level deflection at center of mass"] + df["level edge 1 y deflection from ccw rotation"]
    )
    df["total edge 2 y deflection from ccw rotation"] = (
        df["level deflection at center of mass"] + df["level edge 2 y deflection from ccw rotation"]
    )
    df["total edge 1 x deflection from cw rotation"] = (
        df["level deflection at center of mass"] + df["level edge 1 x deflection from cw rotation"]
    )
    df["total edge 2 x deflection from cw rotation"] = (
        df["level deflection at center of mass"] + df["level edge 2 x deflection from cw rotation"]
    )
    df["total edge 1 y deflection from cw rotation"] = (
        df["level deflection at center of mass"] + df["level edge 1 y deflection from cw rotation"]
    )
    df["total edge 2 y deflection from cw rotation"] = (
        df["level deflection at center of mass"] + df["level edge 2 y deflection from cw rotation"]
    )

    df["average level x deflection from ccw rotation"] = (
        df["total edge 1 x deflection from ccw rotation"] + df["total edge 2 x deflection from ccw rotation"]
    ) / 2
    df["average level y deflection from ccw rotation"] = (
        df["total edge 1 y deflection from ccw rotation"] + df["total edge 2 y deflection from ccw rotation"]
    ) / 2
    df["average level x deflection from cw rotation"] = (
        df["total edge 1 x deflection from cw rotation"] + df["total edge 2 x deflection from cw rotation"]
    ) / 2
    df["average level y deflection from cw rotation"] = (
        df["total edge 1 y deflection from cw rotation"] + df["total edge 2 y deflection from cw rotation"]
    ) / 2

    return df


def _calculate_torsional_irregularity_ratio(shear_walls_dataframe: pd.DataFrame) -> pd.DataFrame:
    df = shear_walls_dataframe
    df["maximum x deflection"] = df[
        [
            "total edge 1 x deflection from ccw rotation",
            "total edge 2 x deflection from ccw rotation",
            "total edge 1 x deflection from cw rotation",
            "total edge 2 x deflection from cw rotation",
        ]
    ].max(axis=1)
    df["maximum y deflection"] = df[
        [
            "total edge 1 y deflection from ccw rotation",
            "total edge 2 y deflection from ccw rotation",
            "total edge 1 y deflection from cw rotation",
            "total edge 2 y deflection from cw rotation",
        ]
    ].max(axis=1)
    df["maximum average x deflection"] = df[
        [
            "average level x deflection from ccw rotation",
            "average level x deflection from cw rotation",
        ]
    ].max(axis=1)
    df["maximum average y deflection"] = df[
        [
            "average level y deflection from ccw rotation",
            "average level y deflection from cw rotation",
        ]
    ].max(axis=1)
    df["torsional irregularity ratio x"] = df["maximum x deflection"] / df["maximum average x deflection"]
    df["torsional irregularity ratio y"] = df["maximum y deflection"] / df["maximum average y deflection"]

    df["torsional irregularity ratio"] = df[
        [
            "torsional irregularity ratio x",
            "torsional irregularity ratio y",
        ]
    ].max(axis=1)

    df["accidental torsional amplification factor x"] = (
        df["maximum x deflection"] / 1.2 / df["maximum average x deflection"]
    ) ** 2
    df["accidental torsional amplification factor y"] = (
        df["maximum y deflection"] / 1.2 / df["maximum average y deflection"]
    ) ** 2

    df["accidental torsional amplification factor x"] = np.where(
        df["accidental torsional amplification factor x"] < 1, 1, df["accidental torsional amplification factor x"]
    )
    df["accidental torsional amplification factor y"] = np.where(
        df["accidental torsional amplification factor y"] < 1, 1, df["accidental torsional amplification factor y"]
    )
    df["accidental torsional amplification factor x"] = np.where(
        df["accidental torsional amplification factor x"] > 3, 3, df["accidental torsional amplification factor x"]
    )
    df["accidental torsional amplification factor y"] = np.where(
        df["accidental torsional amplification factor y"] > 3, 3, df["accidental torsional amplification factor y"]
    )

    return df


def _calculate_inelastic_story_drift(
    shear_walls_dataframe: pd.DataFrame, c_d_x: float, c_d_y: float, i_e: float
) -> pd.DataFrame:
    df = shear_walls_dataframe
    df["story drift at center of mass"] = df.groupby(["Wall"])["level deflection at center of mass"].diff().fillna(0)
    df["edge 1 x story drift from ccw rotation"] = (
        df.groupby(["Wall"])["total edge 1 x deflection from ccw rotation"].diff().fillna(0)
    )
    df["edge 2 x story drift from ccw rotation"] = (
        df.groupby(["Wall"])["total edge 2 x deflection from ccw rotation"].diff().fillna(0)
    )
    df["edge 1 x story drift from cw rotation"] = (
        df.groupby(["Wall"])["total edge 1 x deflection from cw rotation"].diff().fillna(0)
    )
    df["edge 2 x story drift from cw rotation"] = (
        df.groupby(["Wall"])["total edge 2 x deflection from cw rotation"].diff().fillna(0)
    )
    df["edge 1 y story drift from ccw rotation"] = (
        df.groupby(["Wall"])["total edge 1 y deflection from ccw rotation"].diff().fillna(0)
    )
    df["edge 2 y story drift from ccw rotation"] = (
        df.groupby(["Wall"])["total edge 2 y deflection from ccw rotation"].diff().fillna(0)
    )
    df["edge 1 y story drift from cw rotation"] = (
        df.groupby(["Wall"])["total edge 1 y deflection from cw rotation"].diff().fillna(0)
    )
    df["edge 2 y story drift from cw rotation"] = (
        df.groupby(["Wall"])["total edge 2 y deflection from cw rotation"].diff().fillna(0)
    )
    df["maximum edge x story drift"] = df[
        [
            "edge 1 x story drift from ccw rotation",
            "edge 2 x story drift from ccw rotation",
            "edge 1 x story drift from cw rotation",
            "edge 2 x story drift from cw rotation",
        ]
    ].max(axis=1)
    df["maximum edge y story drift"] = df[
        [
            "edge 1 y story drift from ccw rotation",
            "edge 2 y story drift from ccw rotation",
            "edge 1 y story drift from cw rotation",
            "edge 2 y story drift from cw rotation",
        ]
    ].max(axis=1)
    df["elastic story drift x"] = np.where(
        df["torsional irregularity ratio x"] <= 1.2,
        df["story drift at center of mass"],
        df["maximum edge x story drift"],
    )
    df["elastic story drift y"] = np.where(
        df["torsional irregularity ratio x"] <= 1.2,
        df["story drift at center of mass"],
        df["maximum edge x story drift"],
    )
    df["inelastic story drift x"] = df["elastic story drift x"] * c_d_x / i_e
    df["inelastic story drift y"] = df["elastic story drift y"] * c_d_y / i_e

    return df


def _calculate_allowable_story_drift(
    shear_walls_dataframe: pd.DataFrame,
    allowable_story_drift_coefficient: float = 0.020,
    unit_conversion_factor: float = 1,
) -> pd.DataFrame:
    """make this into an automatic table reference at some point

    Args:
        shear_walls_dataframe (pd.DataFrame): _description_
        unit_conversion_factor (float, optional): _description_. Defaults to 12.

    Returns:
        pd.DataFrame: _description_
    """
    df = shear_walls_dataframe
    df["allowable story drift"] = df["wall height"] * allowable_story_drift_coefficient * unit_conversion_factor
    return df


def calculate_deflections(
    shear_walls_dataframe: pd.DataFrame,
    plan_dimensions: tuple[float, float],
    c_d_x: float,
    c_d_y: float,
    i_e: float,
    allowable_story_drift_coefficient: float = 0.020,
    unit_conversion_factor: float = 1,
) -> pd.DataFrame:
    df = shear_walls_dataframe
    df = _calculate_deflection_at_center_of_mass(df)
    df = _calculate_rotation_at_center_of_rigidity(df)
    df = _calculate_deflection_at_diaphragm_edges_from_rotation(df, plan_dimensions)
    df = _calculate_total_deflection_at_diaphragm_edges(df)
    df = _calculate_torsional_irregularity_ratio(df)
    df = _calculate_inelastic_story_drift(df, c_d_x, c_d_y, i_e)
    df = _calculate_allowable_story_drift(df, allowable_story_drift_coefficient, unit_conversion_factor)
    return df


def calculate_required_strap_capacity_interior_walls(
    shear_walls_dataframe: pd.DataFrame, nominal_diaphragm_capacity: float
) -> pd.DataFrame:
    """THIS ASSUMES DIAPHRAGM ON BOTH SIDES OF THE SHEAR WALL. THIS DOES NOT WORK FOR EDGE WALLS.

    FURTHER IMPLEMENTATION IS NEEDED

    Args:
        shear_walls_dataframe (pd.DataFrame): _description_
        nominal_diaphragm_capacity (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = shear_walls_dataframe
    df["required strap capacity"] = (
        df["line unit shear"] * df["diaphragm force scale factor"] * df["line wall length"]
        - 2 * nominal_diaphragm_capacity * SPDWS_LOAD_CASE_FACTOR_SEISMIC_ASD * df["line wall length"]
    )
    df["required strap capacity"] = np.where(df["required strap capacity"] < 0, 0, df["required strap capacity"])
    return df


def calculate_required_strap_capacity_exterior_walls(
    shear_walls_dataframe: pd.DataFrame, nominal_diaphragm_capacity: float
) -> pd.DataFrame:
    """THIS ASSUMES DIAPHRAGM ON ONE SIDE OF THE SHEAR WALL. THIS DOES NOT WORK FOR INTERIOR WALLS.

    FURTHER IMPLEMENTATION IS NEEDED

    Args:
        shear_walls_dataframe (pd.DataFrame): _description_
        nominal_diaphragm_capacity (float): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df = shear_walls_dataframe
    df["required strap capacity"] = (
        df["adjusted diaphragm floor unit shear demand"] * df["wall length"]
        - nominal_diaphragm_capacity * SPDWS_LOAD_CASE_FACTOR_SEISMIC_ASD * df["wall length"]
    )
    df["required strap capacity"] = np.where(df["required strap capacity"] < 0, 0, df["required strap capacity"])
    return df


# def recalculate_shear_demand_with_accidental_moment_amplification
