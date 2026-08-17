from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum

import numpy as np
import requests

from .. import ASCECodeVersion
from .general import RiskCategory

LOAD_FACTOR_SEISMIC_ASD = 0.7
LOAD_FACTOR_SEISMIC_LRFD = 1.0


class SiteClass(Enum):
    A = "A"
    B = "B"
    BC = "BC"
    C = "C"
    CD = "CD"
    D = "D"
    DE = "DE"
    E = "E"
    F = "F"

    @property
    def severity(self):
        order = {
            SiteClass.A: 1,
            SiteClass.B: 2,
            SiteClass.BC: 2.5,
            SiteClass.C: 3,
            SiteClass.CD: 3.5,
            SiteClass.D: 4,
            SiteClass.DE: 4.5,
            SiteClass.E: 5,
            SiteClass.F: 6,
        }
        return order[self]

    def __lt__(self, other):
        return self.severity < other.severity


class SeismicDesignCategory(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"

    @property
    def severity(self):
        order = {
            SeismicDesignCategory.A: 1,
            SeismicDesignCategory.B: 2,
            SeismicDesignCategory.C: 3,
            SeismicDesignCategory.D: 4,
            SeismicDesignCategory.E: 5,
            SeismicDesignCategory.F: 6,
        }
        return order[self]

    def __lt__(self, other):
        return self.severity < other.severity


# ============================================================================
# ASCE 7-16 TABLES
# ============================================================================

SEISMIC_IMPORTANCE_FACTORS: dict[RiskCategory, float] = {
    RiskCategory.I: 1.0,
    RiskCategory.II: 1.0,
    RiskCategory.III: 1.25,
    RiskCategory.IV: 1.5,
}

F_A_TABLE_11_4_1: dict[SiteClass, dict[str, list[float]]] = {
    SiteClass.A: {
        "s_s": [0.25, 0.50, 0.75, 1.00, 1.25, 1.5],
        "f_a": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    },
    SiteClass.B: {
        "s_s": [0.25, 0.50, 0.75, 1.00, 1.25, 1.5],
        "f_a": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
    },
    SiteClass.C: {
        "s_s": [0.25, 0.50, 0.75, 1.00, 1.25, 1.5],
        "f_a": [1.3, 1.3, 1.2, 1.2, 1.2, 1.2],
    },
    SiteClass.D: {
        "s_s": [0.25, 0.50, 0.75, 1.00, 1.25, 1.5],
        "f_a": [1.6, 1.4, 1.2, 1.1, 1.0, 1.0],
    },
    SiteClass.E: {
        "s_s": [0.25, 0.50, 0.75, 1.00, 1.25, 1.5],
        "f_a": [2.4, 1.7, 1.3, np.nan, np.nan, np.nan],
    },
    SiteClass.F: {
        "s_s": [0.25, 0.50, 0.75, 1.00, 1.25, 1.5],
        "f_a": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    },
}

F_V_TABLE_11_4_2: dict[SiteClass, dict[str, list[float]]] = {
    SiteClass.A: {
        "s_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "f_v": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    },
    SiteClass.B: {
        "s_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "f_v": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
    },
    SiteClass.C: {
        "s_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "f_v": [1.5, 1.5, 1.5, 1.5, 1.5, 1.4],
    },
    SiteClass.D: {
        "s_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "f_v": [2.4, 2.2, 2.0, 1.9, 1.8, 1.7],
    },
    SiteClass.E: {
        "s_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "f_v": [4.2, np.nan, np.nan, np.nan, np.nan, np.nan],
    },
    SiteClass.F: {
        "s_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "f_v": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    },
}

C_U_TABLE_12_8_1: dict[str, list[float]] = {
    "s_d1": [0.1, 0.15, 0.2, 0.3, 0.4],
    "c_u": [1.7, 1.6, 1.5, 1.4, 1.4],
}

# ============================================================================
# GENERIC INTERPOLATION UTILITIES
# ============================================================================


def _linear_interpolate(
    x: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """
    Perform straight-line interpolation between two points.
    """

    if x2 == x1:
        raise ValueError("Interpolation points must have different x values.")

    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)


def _interpolate_table(
    x: float,
    x_values: list[float],
    y_values: list[float],
) -> float:
    """
    Interpolate a value from tabulated data.

    Parameters
    ----------
    x : float
        Input x value.

    x_values : list[float]
        Ordered x-axis table values.

    y_values : list[float]
        Corresponding y-axis table values.

    Returns
    -------
    float
        Interpolated value.
    """
    if np.isnan(y_values):
        raise ValueError(
            "You are trying to interpolate values that are NAN. This means you are in a regime of the table that does not have values according to ASCE. See the relevant table to see where values are not defined."
        )

    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have equal length.")

    if sorted(x_values) != x_values:
        raise ValueError("x_values must be sorted ascending.")

    # Clamp below table range
    if x <= x_values[0]:
        return y_values[0]

    # Clamp above table range
    if x >= x_values[-1]:
        return y_values[-1]

    idx = bisect_right(x_values, x)

    if y_values[idx - 1] is None or y_values[idx] is None:
        raise ValueError(
            "This combination of Spectral Response Acceleration Parameter and Site Class requires that you see ASCE 7-16 Section 11.4.8."
        )

    return _linear_interpolate(
        x=x,
        x1=x_values[idx - 1],
        y1=y_values[idx - 1],
        x2=x_values[idx],
        y2=y_values[idx],
    )


# ============================================================================
# SEISMIC PARAMETERS DATACLASS
# ============================================================================


@dataclass
class SeismicParameters:
    """
    Encapsulates seismic parameters for a given site and spectral response acceleration.

    Note that 0.001 degrees of latitude and longitude is on the order of about 100 meters, so you only need 3-4 decimals of precision for lat/long.
    """

    site_class: SiteClass = SiteClass.D
    risk_category: RiskCategory = RiskCategory.II
    asce_code_version: ASCECodeVersion = ASCECodeVersion.ASCE_7_22
    latitude: float | None = None
    longitude: float | None = None
    s_s: float | None = None
    s_1: float | None = None
    t_l: float = 16
    request_timeout: int = 10

    def __post_init__(self):
        self.i_e = SEISMIC_IMPORTANCE_FACTORS[self.risk_category]
        if self.asce_code_version is ASCECodeVersion.ASCE_7_22:
            if not self.latitude or not self.longitude:
                raise ValueError(
                    "Supply Latitude and Longitude to the class constructor to obtain seismic parameters using the USGS API endpoint (same as ASCE Hazard Tool)"
                )
            self.site_data = self.get_site_seismic_values()
            self.pga_m = self.site_data["pgam"]
            self.s_ms = self.site_data["sms"]
            self.s_m1 = self.site_data["sm1"]
            self.s_ds = self.site_data["sds"]
            self.s_d1 = self.site_data["sd1"]
            self.sdc = self.site_data["sdc"]
            self.s_s = self.site_data["ss"]
            self.s_1 = self.site_data["s1"]
            self.t_l = self.site_data["tl"]
            self.t_s = self.site_data["ts"]
            self.t_0 = self.site_data["t0"]
            self.c_v = self.site_data["cv"]
        else:
            if not self.s_s or self.s_1:
                raise ValueError(
                    "Supply s_1 and s_s to the class constructor to obtain seismic parameters using ASCE 7-16 Section 11."
                )
            self.f_a = self._get_f_a_coefficient()
            self.f_v = self._get_f_v_coefficient()

            self.s_ms = self.s_s * self.f_a
            self.s_m1 = self.s_1 * self.f_v

            self.s_ds = 2 / 3 * self.s_ms
            self.s_d1 = 2 / 3 * self.s_m1

            self.c_u = self._get_c_u_coefficient()

            self.sdc = self._get_seismic_design_category()

    def get_site_seismic_values(self) -> dict:
        response = requests.get(
            "https://earthquake.usgs.gov/ws/building-codes/asce7-22/calculate",
            params={
                "latitude": self.latitude,
                "longitude": self.longitude,
                "siteClass": self.site_class.value,
                "riskCategory": self.risk_category.value,
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.json()["response"]["data"]

    def _get_f_a_coefficient(self) -> float:
        """
        Compute F_a from ASCE 7-16 Table 11.4-1.

        Parameters
        ----------
        s_s : float
            Mapped spectral acceleration parameter at short periods.

        site_class : str
            Site class designation ('A', 'B', 'C', 'D', 'E', or 'F').

        Returns
        -------
        float
            Interpolated F_a coefficient.
        """

        table = F_A_TABLE_11_4_1[self.site_class]

        return _interpolate_table(
            x=self.s_s,
            x_values=table["s_s"],
            y_values=table["f_a"],
        )

    def _get_f_v_coefficient(self) -> float:
        """
        Compute F_v from ASCE 7-16 Table 11.4-2.

        Parameters
        ----------
        s_1 : float
            Mapped spectral acceleration parameter at 1.0 second period.

        site_class : str
            Site class designation ('A', 'B', 'C', 'D', 'E', or 'F').

        Returns
        -------
        float
            Interpolated F_v coefficient.
        """

        table = F_V_TABLE_11_4_2[self.site_class]

        return _interpolate_table(
            x=self.s_1,
            x_values=table["s_1"],
            y_values=table["f_v"],
        )

    def _get_c_u_coefficient(self) -> float:
        """
        Compute C_u from ASCE 7-16 Table 12.8-1.

        Parameters
        ----------
        s_d1 : float
            Design spectral acceleration parameter at 1.0 second period.

        Returns
        -------
        float
            Interpolated C_u coefficient.
        """

        table = C_U_TABLE_12_8_1

        return _interpolate_table(
            x=self.s_d1,
            x_values=table["s_d1"],
            y_values=table["c_u"],
        )

    def _get_seismic_design_category(self) -> str:
        """
        Determine seismic design category from ASCE 7-16 Table 11.6-1 or 11.6-2.

        Returns
        -------
        SeismicDesignCategory
            Seismic design category.
        """
        SDC = SeismicDesignCategory
        is_risk_iv = self.risk_category == RiskCategory.IV

        # S_1 >= 0.75 check takes precedence per Tables 11.6-1/2
        if self.s_1 >= 0.75:
            return (SDC.F if is_risk_iv else SDC.E).name

        # Thresholds and categories per Table 11.6-1 (I/II/III) and 11.6-2 (IV)
        sds_thresholds = [
            (0.167, SDC.A),
            (0.33, SDC.B if not is_risk_iv else SDC.C),
            (0.50, SDC.C if not is_risk_iv else SDC.D),
            (float("inf"), SDC.D),
        ]
        sd1_thresholds = [
            (0.067, SDC.A),
            (0.133, SDC.B if not is_risk_iv else SDC.C),
            (0.20, SDC.C if not is_risk_iv else SDC.D),
            (float("inf"), SDC.D),
        ]

        def lookup(value: float, thresholds: list) -> SeismicDesignCategory:
            return next(category for limit, category in thresholds if value < limit)

        sdc_short = lookup(self.s_ds, sds_thresholds)
        sdc_long = lookup(self.s_d1, sd1_thresholds)

        return max(sdc_short, sdc_long).name
