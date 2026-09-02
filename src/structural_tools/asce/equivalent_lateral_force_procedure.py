from dataclasses import dataclass

import numpy as np
import pandas as pd

from structural_tools.structure import Structure

from .seismic_parameters import SeismicParameters

LOWER_BOUND_C_S_CHECK_S_1 = 0.6


@dataclass
class SeismicLoads:
    """Seismic loading class containing individual parameters and the full seismic loads table as a pd.DataFrame.

    The returned columns are:
        "level height": "$h_{floor}$ [ft]",
        "level weight": "$w_x$ [kip]",
        "level elevation": "$h_x$ [ft]",
        "level weighting parameter": "$w_x h_x^k$",
        "vertical distribution factor": "$C_{vx}$",
        "lateral seismic force": "$F_x$ [kip]",
        "cumulative lateral seismic force": r"$\sum F_x$ [kip]",
        "unbounded diaphragm design force": "$F_{px,u}$ [kip]",
        "maximum diaphragm design force": "$F_{px,max}$ [kip]",
        "diaphragm design force": "$F_{px}$ [kip]",
        "seismic design story shear": "$V_x$ [kip]",
        "overturning moment": "OTM [kip-ft]",

    Raises:
        ValueError: Building period must be greater than 0
    """

    structure: Structure
    seismic_parameters: SeismicParameters

    def __post_init__(self):
        self.base_shear_x = self._calculate_base_shear(self.c_s_x)
        self.base_shear_y = self._calculate_base_shear(self.c_s_y)
        self.v_x = self.base_shear_x
        self.v_y = self.base_shear_y
        self.seismic_loads_x = self._calculate_seismic_loads(
            self.base_shear_x, self._k_x, self.structure.plan_dimensions[1]
        )
        self.seismic_loads_y = self._calculate_seismic_loads(
            self.base_shear_y, self._k_y, self.structure.plan_dimensions[0]
        )

    @property
    def _k_x(self) -> float:
        return self._calculate_structural_period_exponent(self.structure.period_x)

    @property
    def _k_y(self) -> float:
        return self._calculate_structural_period_exponent(self.structure.period_y)

    @property
    def c_s_x(self) -> float:
        return self._calculate_seismic_response_coefficient(self.structure.period_x, self.structure.lateral_system_x.r)

    @property
    def c_s_y(self) -> float:
        return self._calculate_seismic_response_coefficient(self.structure.period_y, self.structure.lateral_system_y.r)

    @property
    def column_name_map(self) -> dict[str, str]:
        return {
            "level height": "$h_{floor}$ [ft]",
            "level weight": "$w_x$ [kip]",
            "level elevation": "$h_x$ [ft]",
            "level weighting parameter": "$w_x h_x^k$",
            "vertical distribution factor": "$C_{vx}$",
            "lateral seismic force": "$F_x$ [kip]",
            "cumulative lateral seismic force": r"$\sum F_x$ [kip]",
            "unbounded diaphragm design force": "$F_{px,u}$ [kip]",
            "maximum diaphragm design force": "$F_{px,max}$ [kip]",
            "diaphragm design force": "$F_{px}$ [kip]",
            "seismic design story shear": "$V_x$ [kip]",
            "overturning moment": "OTM [kip-ft]",
        }

    # Uses method 2 of ASCE7-22 Sec. 12.8.1.1
    def _calculate_seismic_response_coefficient(self, T, R) -> float:
        p = self.seismic_parameters
        C_s = p.s_ds / (R / p.i_e)
        if T <= p.t_l:
            C_s_max = p.s_d1 / (T * R / p.i_e)
        else:
            C_s_max = p.s_d1 * p.t_l / (T**2 * R / p.i_e)

        C_s_min = max(0.044 * p.s_ds * p.i_e, 0.01)

        if p.s_1 >= LOWER_BOUND_C_S_CHECK_S_1:
            C_s_min = max(C_s_min, 0.5 * p.s_1 / (R / p.i_e))

        return max(C_s_min, min(C_s, C_s_max))

    def _calculate_base_shear(self, c_s) -> float:
        return c_s * self.structure.effective_seismic_weight

    def _calculate_structural_period_exponent(self, T: float) -> float:
        """Given the building period, calculate the structural period exponent from ASCE 7-16 Section 12.8.

        Args:
            T (float): Building period in seconds

        Raises:
            ValueError: If the building period is <= 0

        Returns:
            float: Structural period exponent
        """
        if T <= 0:
            raise ValueError(f"Building period must be greater than zero. Your building period is {T}")

        if T <= 0.5:
            k = 1
        elif T >= 2.5:
            k = 2
        else:
            k = 1 + ((T - 0.5) * (2 - 1) / (2.5 - 0.5))

        return k

    def _calculate_seismic_loads(self, V, k, l_diaph, r_diaph=4.5) -> pd.DataFrame:
        p = self.seismic_parameters
        # Generate seismic loads DataFrame
        seismic_loads = self.structure.levels_data.copy()
        # Calc cumulative level weight
        seismic_loads["cumulative level weight"] = seismic_loads.iloc[::-1]["level weight"].cumsum()

        # Calc coefficients and story forces ASCE 7-16 Eq. 12.8-11 and 12.8-12
        seismic_loads["level weighting parameter"] = seismic_loads["level weight"] * seismic_loads[
            "level elevation"
        ] ** float(k)
        cumulative_level_weighting_parameter = seismic_loads["level weighting parameter"].sum()
        seismic_loads["vertical distribution factor"] = (
            seismic_loads["level weighting parameter"] / cumulative_level_weighting_parameter
        )
        seismic_loads["lateral seismic force"] = seismic_loads["vertical distribution factor"] * float(V)

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

        # Calc diaphragm loads
        seismic_loads["cumulative lateral seismic force"] = seismic_loads.iloc[::-1]["lateral seismic force"].cumsum()
        seismic_loads["unbounded diaphragm design force"] = p.s_ds / r_diaph * p.i_e * seismic_loads["level weight"]
        # set bounds on diaphragm loads
        seismic_loads["maximum diaphragm design force"] = (
            p.s_d1 / (0.002 * l_diaph) / r_diaph * p.i_e * seismic_loads["level weight"]
        )
        # bound the diaphragm design force
        seismic_loads["diaphragm design force"] = np.where(
            seismic_loads["unbounded diaphragm design force"] >= seismic_loads["maximum diaphragm design force"],
            seismic_loads["maximum diaphragm design force"],
            seismic_loads["unbounded diaphragm design force"],
        )

        return seismic_loads
