from enum import Enum


class DesignMethod(Enum):
    ASD = "ASD"
    LRFD = "LRFD"

class CodeVersion(Enum):
    AISC_360_16 = 2016
    AISC_360_18 = 2018
    AISC_360_22 = 2022
    ASCE_7_16 = 2016
    ASCE_7_22 = 2022
    NDS_2015 = 2015
    NDS_2018 = 2018
    NDS_2024 = 2024
    SDPWS_2015 = 2015
    SDPWS_2021 = 2021
