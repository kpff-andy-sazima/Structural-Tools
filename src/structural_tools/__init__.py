from .notebook import initialize_notebook
from enum import Enum


class DesignMethod(Enum):
    ASD = "ASD"
    LRFD = "LRFD"


class ASCECodeVersion(Enum):
    ASCE_7_16 = 2016
    ASCE_7_22 = 2022


class ACICodeVersion(Enum):
    ACI_318_14 = 2014
    ACI_318_19 = 2019
    ACI_318_25 = 2025


class AISCCodeVersion(Enum):
    AISC_360_16 = 2016
    AISC_360_18 = 2018
    AISC_360_22 = 2022


class NDSCodeVersion(Enum):
    NDS_2015 = 2015
    NDS_2018 = 2018
    NDS_2024 = 2024


class SDPWSCodeVersion(Enum):
    SDPWS_2015 = 2015
    SDPWS_2021 = 2021


class LoadCase(Enum):
    """ASCE 7 principal load cases. Value is the primary PyNite case label.

    SEISMIC spans two labels (E_v, E_h); the value is informational only since
    inclusion is driven by the enum member, not the label string.
    """

    DEAD = "D"
    LIVE = "L"
    LIVE_ROOF = "L_r"
    SNOW = "S"
    RAIN = "R"
    WIND = "W"
    SEISMIC = "E"


__all__ = ["DesignMethod", "ASCECodeVersion", "initialize_notebook"]
