import pandas as pd
import numpy as np
from forallpeople import Physical
from ..util import generate_levels_list


class SeismicLoads:
    levels: list[float]
    level_heights: list[float]
    level_weights: list[float]
    base_shear: float
    building_period: float
    add_roof: bool = True
    reverse_level_order: bool = True
    s_ds: float | None = None
    i_e: float | None = None


def get_structural_period_exponent(building_period: float) -> float:
    """Given the building period, calculate the structural period exponent from ASCE 7-16 Section 12.8.

    Args:
        building_period (float): Building period in seconds

    Raises:
        ValueError: If the building period is <= 0

    Returns:
        float: Structural period exponent
    """
    if building_period <= 0:
        raise ValueError(f"Building period must be greater than zero. Your building period is {building_period}")

    if building_period <= 0.5:
        k = 1
    elif building_period >= 2.5:
        k = 2
    else:
        k = 1 + ((building_period - 0.5) * (2 - 1) / (2.5 - 0.5))

    return k


def generate_seismic_loads(
    base_shear: float | Physical,
    level_heights: list[float | Physical],
    level_weights: list[float | Physical],
    number_of_levels: int | None = None,
    levels_list: list[str] | None = None,
    structural_period_exponent: float = 1,
    add_roof: bool = True,
    reverse_order: bool = True,
    s_ds: float | None = None,
    i_e: float | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Generate a building's seismic loading for each level. If s_ds and i_e are supplied, the diaphragm forces will be automatically bounded based on ASCE 7 Section 12.10

    Args:
        levels_list (list[str]): The names of each level. Recommended to generate this with 'generate_levels_list()', but can be any list of strings.
        level_heights (list[float  |  Physical]): Each level's height. If a roof is present (with 'add_roof=True'), the final entry is expected to be 0. If not, a 0 is added.
        level_weights (list[float  |  Physical]): Each level's weight. The first level is expected to have 0 seismic weight; if the first entry is not 0, it is naively assumed that you started the weights with the second level, so a 0 is inserted at the front.
        base_shear (float | Physical): The base shear of the building calculated with ASCE 7 Section 12.8.1.
        structural_period_exponent (float, optional): The exponent 'k' used in the calculation of the vertical distribution factor. Defaults to 1.
        add_roof (bool, optional): Option to add a roof level to the levels. Defaults to True.
        reverse_order (bool, optional): Should the final DataFrame order be reversed (top level first, ground level last). Defaults to True.
        s_ds (float | None, optional): Design spectral response acceleration, S_DS, from ASCE 7 Section 11.4. Defaults to None.
        i_e (float | None, optional): Importance factor, I_e, from ASCE 7 Section 11.5 and 1.5. Defaults to None.

    Raises:
        ValueError: If neither 'number_of_levels' or 'levels_list' are supplied.
        ValueError: If both 'number_of_levels' and 'levels_list' are supplied.
        ValueError: If levels_list and level_heights are not the same length.
        ValueError: If levels_list and level_heights are not the same length.

    Returns:
        pd.DataFrame: Seismic forces
        dict[str,str]: Mapping of series names and common LaTeX table headers
    """
    if not number_of_levels and not levels_list:
        raise ValueError("At least one of 'number_of_levels' or 'levels_list' should be supplied.")

    if number_of_levels and levels_list:
        raise ValueError("Supply only one of 'number_of_levels' or 'levels_list'.")

    if number_of_levels and not levels_list:
        levels_list = generate_levels_list(number_of_levels=number_of_levels, add_roof=add_roof)

    # Roof should have 0 height. Naively assume that the user just didn't include a zero at the end
    if add_roof and level_heights[-1] != 0:
        level_heights.append(0)

    # Level 1 should have 0 weight. Naively assume that the user just didn't include a zero at the start
    if level_weights[0] != 0:
        level_weights.insert(0, 0)

    # Check lengths of lists to ensure they match
    if levels_list and len(levels_list) != len(level_heights):
        raise ValueError(
            f"Each level needs one height.\n\
            Your levels list has {len(levels_list)} entries:\n\
            {levels_list}\n\
            Your level heights list las {len(level_heights)} entries:\n\
            {level_heights}"
        )
    if levels_list and len(levels_list) != len(level_weights):
        raise ValueError(
            f"Each level needs one weight.\n\
            Your levels list has {len(levels_list)} entries:\n\
            {levels_list}\n\
            Your level weights list las {len(level_weights)} entries:\n\
            {level_weights}"
        )

    # Convert Physical values to floats
    level_heights = [float(v) if isinstance(v, Physical) else v for v in level_heights]
    level_weights = [float(v) if isinstance(v, Physical) else v for v in level_weights]
    if isinstance(base_shear, Physical):
        base_shear = float(base_shear)

    # Generate seismic loads DataFrame
    seismic_loads = pd.DataFrame({
        "Level": levels_list,
        "level height": level_heights,
        "level weight": level_weights,
    })

    seismic_loads = seismic_loads.set_index("Level")
    # Calc cumulative level weight
    seismic_loads["cumulative level weight"] = seismic_loads.iloc[::-1]["level weight"].cumsum()

    # Calc cumulative elevation
    elevations = [0]
    for height in seismic_loads["level height"][:-1]:
        elevations.append(height + elevations[-1])
    seismic_loads.insert(2, "level elevation", elevations)

    # Calc coefficients and story forces ASCE 7-16 Eq. 12.8-11 and 12.8-12
    seismic_loads["level weighting parameter"] = seismic_loads["level weight"] * seismic_loads[
        "level elevation"
    ] ** float(structural_period_exponent)
    cumulative_level_weighting_parameter = seismic_loads["level weighting parameter"].sum()
    seismic_loads["vertical distribution factor"] = (
        seismic_loads["level weighting parameter"] / cumulative_level_weighting_parameter
    )
    seismic_loads["lateral seismic force"] = seismic_loads["vertical distribution factor"] * float(base_shear)
    seismic_loads["cumulative lateral seismic force"] = seismic_loads.iloc[::-1]["lateral seismic force"].cumsum()
    seismic_loads["unbounded diaphragm design force"] = (
        seismic_loads["cumulative lateral seismic force"]
        / seismic_loads["cumulative level weight"]
        * seismic_loads["level weight"]
    )
    if s_ds and i_e:
        # set bounds
        seismic_loads["minimum diaphragm design force"] = 0.2 * s_ds * i_e * seismic_loads["level weight"]
        seismic_loads["maximum diaphragm design force"] = 0.4 * s_ds * i_e * seismic_loads["level weight"]
        # bound the diaphragm design force
        seismic_loads["diaphragm design force"] = np.where(
            seismic_loads["unbounded diaphragm design force"] <= seismic_loads["minimum diaphragm design force"],
            seismic_loads["minimum diaphragm design force"],
            np.where(
                seismic_loads["unbounded diaphragm design force"] >= seismic_loads["maximum diaphragm design force"],
                seismic_loads["maximum diaphragm design force"],
                seismic_loads["unbounded diaphragm design force"],
            ),
        )

    # Calc story shear forces ASCE 7-16 Eq. 12.8-13
    story_shears = [0]
    for shear in reversed(list(seismic_loads["lateral seismic force"][1:])):
        story_shears.insert(0, shear + story_shears[0])
    seismic_loads["seismic design story shear"] = story_shears

    # Calc overturning moments
    floor_moments = seismic_loads["seismic design story shear"] * seismic_loads["level height"]
    overturning_moments = [0]
    for floor_moment in reversed(list(floor_moments[:-1])):
        overturning_moments.insert(0, overturning_moments[0] + floor_moment)
    seismic_loads["overturning moment"] = overturning_moments

    # Calculate bounded diaphragm forces if s_ds and i_e are supplied

    if reverse_order:
        seismic_loads = seismic_loads.iloc[::-1]

    column_name_map = {
        "level height": "$h_{floor}$ [ft]",
        "level weight": "$w_x$ [kip]",
        "level elevation": "$h_x$ [ft]",
        "level weighting parameter": "$w_x h_x^k$",
        "vertical distribution factor": "$C_{vx}$",
        "lateral seismic force": "$F_x$ [kip]",
        "cumulative lateral seismic force": r"$\sum F_x$ [kip]",
        "unbounded diaphragm design force": "$F_{px,u}$ [kip]",
        "minimum diaphragm design force": "$F_{px,min}$ [kip]",
        "maximum diaphragm design force": "$F_{px,max}$ [kip]",
        "diaphragm design force": "$F_{px}$ [kip]",
        "seismic design story shear": "$V_x$ [kip]",
        "overturning moment": "OTM [kip-ft]",
    }

    return seismic_loads, column_name_map
