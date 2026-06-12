import pandas as pd
from .util import generate_levels_list
from pathlib import Path


def _set_shear_wall_heights(
    shear_walls_dataframe: pd.DataFrame, shear_wall_heights: pd.Series | pd.DataFrame
) -> pd.DataFrame:
    # Cleansing shear wall heights
    if isinstance(shear_wall_heights, (pd.Series, pd.DataFrame)) and "Roof" in shear_wall_heights.index:
        shear_wall_heights = shear_wall_heights.drop("Roof")
    if isinstance(shear_wall_heights, pd.DataFrame) and "level height" in shear_wall_heights.columns:
        shear_wall_heights = shear_wall_heights["level height"]

    df = shear_walls_dataframe

    df["wall height"] = shear_wall_heights
    return df


def create_shear_walls_dataframe_from_excel(
    excel_workbook: Path | str,
    number_of_levels: int,
    shear_wall_heights: pd.Series | pd.DataFrame,
    are_stacking: bool = True,
    worksheet: str = "Input - Stacking Walls",
) -> pd.DataFrame:
    levels = generate_levels_list(number_of_levels=number_of_levels, add_roof=False, reverse_list=True)
    if are_stacking:
        level_df = pd.read_excel(excel_workbook, worksheet)
        level_df.columns = level_df.columns.str.lower()
        level_df = level_df.rename(columns={"wall": "Wall"})
        level_df = level_df.set_index("Wall")
        shear_walls = pd.concat({level: level_df for level in levels}, names=["Level", "Wall"])
    else:
        shear_walls = pd.read_excel("Shear Walls Input.xlsx", "Input - Stacking Walls")
        shear_walls.columns = shear_walls.columns.str.lower()
        shear_walls = shear_walls.rename(columns={"level": "Level", "wall": "Wall"})
        shear_walls = shear_walls.set_index(["Level", "Wall"])

    shear_walls["x"] = shear_walls["x"].fillna(0)
    shear_walls["y"] = shear_walls["y"].fillna(0)
    shear_walls["tributary area"] = shear_walls["tributary area"].fillna(0)
    shear_walls["tributary area"] = shear_walls["tributary area"].where(
        shear_walls["tributary area"] != 0, shear_walls["tributary width"] * shear_walls["tributary length"]
    )
    shear_walls.drop(columns=["tributary length", "tributary width"], inplace=True)

    return shear_walls
