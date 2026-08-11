"""AISC 360-22 Section F4 — other I-shaped members with compact or noncompact
webs, major axis (doubly symmetric with noncompact webs, and singly symmetric
with compact or noncompact webs).

Limit states: compression flange yielding, lateral-torsional buckling,
compression flange local buckling, tension flange yielding.

Units: kip-inch.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import Optional

from ...typing import FloatLike
from .classification import SlendernessCheck
from .common import (
    Axis,
    ElementClass,
    FlexuralStrength,
    LimitState,
    interpolate_noncompact,
    missing,
    missing_note,
)
from ..constants import YOUNGS_MODULUS_KSI
from .properties import SectionProperties


def calculate_limiting_flange_stress(
    yield_stress: FloatLike,
    section_modulus_tension: FloatLike,
    section_modulus_compression: FloatLike,
) -> float:
    """F_L, the stress above which inelastic buckling applies (Eq. F4-6a / F4-6b)."""
    F_y = float(yield_stress)
    ratio = float(section_modulus_tension) / float(section_modulus_compression)
    if ratio >= 0.7:
        return 0.7 * F_y
    return max(F_y * ratio, 0.5 * F_y)


def calculate_web_plastification_factor(
    plastic_moment: FloatLike,
    yield_moment: FloatLike,
    web_slenderness: FloatLike,
    limiting_compact_web: FloatLike,
    limiting_noncompact_web: FloatLike,
    flange_inertia_ratio: FloatLike,
) -> float:
    """Web plastification factor R_pc or R_pt (Eq. F4-9, F4-10, F4-16, F4-17).

    Args:
        plastic_moment: M_p = F_y Z_x <= 1.6 F_y S_x
        yield_moment: M_yc for R_pc, M_yt for R_pt
        web_slenderness: h_c/t_w
        limiting_compact_web: lambda_pw
        limiting_noncompact_web: lambda_rw
        flange_inertia_ratio: I_yc / I_y
    """
    if float(flange_inertia_ratio) <= 0.23:
        return 1.0  # Eq. F4-10 / F4-17
    ratio = float(plastic_moment) / float(yield_moment)
    lam = float(web_slenderness)
    lam_pw = float(limiting_compact_web)
    lam_rw = float(limiting_noncompact_web)
    if lam <= lam_pw:
        return ratio  # Eq. F4-9a / F4-16a
    factor = ratio - (ratio - 1.0) * (lam - lam_pw) / (lam_rw - lam_pw)  # F4-9b / F4-16b
    return min(factor, ratio)


def calculate_limiting_yield_length(
    yield_stress: FloatLike,
    compression_flange_radius: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Limiting unbraced length for yielding, L_p (Eq. F4-7)."""
    return 1.1 * float(compression_flange_radius) * sqrt(
        float(youngs_modulus) / float(yield_stress)
    )


def calculate_limiting_inelastic_length(
    limiting_flange_stress: FloatLike,
    compression_flange_radius: FloatLike,
    torsional_constant: FloatLike,
    section_modulus_compression: FloatLike,
    flange_centroid_distance: FloatLike,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Limiting unbraced length for inelastic LTB, L_r (Eq. F4-8)."""
    F_L = float(limiting_flange_stress)
    r_t = float(compression_flange_radius)
    J = float(torsional_constant)
    S_xc = float(section_modulus_compression)
    h_o = float(flange_centroid_distance)
    E = float(youngs_modulus)
    beta = J / (S_xc * h_o)
    return 1.95 * r_t * (E / F_L) * sqrt(beta + sqrt(beta**2 + 6.76 * (F_L / E) ** 2))


def calculate_elastic_lateral_torsional_buckling_stress(
    unbraced_length: FloatLike,
    compression_flange_radius: FloatLike,
    torsional_constant: FloatLike,
    section_modulus_compression: FloatLike,
    flange_centroid_distance: FloatLike,
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> float:
    """Critical stress F_cr for elastic LTB (Eq. F4-5).

    J must already be set to zero by the caller when I_yc/I_y <= 0.23.
    """
    L_b = float(unbraced_length)
    r_t = float(compression_flange_radius)
    J = float(torsional_constant)
    S_xc = float(section_modulus_compression)
    h_o = float(flange_centroid_distance)
    C_b = float(ltb_modification_factor)
    E = float(youngs_modulus)
    slenderness = L_b / r_t
    return C_b * pi**2 * E / slenderness**2 * sqrt(
        1 + 0.078 * J / (S_xc * h_o) * slenderness**2
    )


def check_f4(
    properties: SectionProperties,
    yield_stress: FloatLike,
    unbraced_length: FloatLike,
    flange: Optional[SlendernessCheck],
    web: Optional[SlendernessCheck],
    ltb_modification_factor: FloatLike = 1.0,
    youngs_modulus: FloatLike = YOUNGS_MODULUS_KSI,
) -> FlexuralStrength:
    """Section F4 check.

    Args:
        properties: derived SectionProperties.
        yield_stress: F_y (ksi).
        unbraced_length: L_b (in).
        flange: Table B4.1b check for the compression flange.
        web: Table B4.1b check for the web.
        ltb_modification_factor: C_b (Eq. F1-1).
        youngs_modulus: E (ksi).
    """
    F_y = float(yield_stress)
    E = float(youngs_modulus)
    C_b = float(ltb_modification_factor)
    L_b = float(unbraced_length)

    result = FlexuralStrength(
        section_reference="F4",
        title="Other I-shaped members with compact or noncompact webs bent about their major axis",
        axis=Axis.MAJOR,
    )
    absent = missing(
        properties,
        [
            "plastic_modulus_x",
            "section_modulus_x",
            "section_modulus_compression",
            "section_modulus_tension",
            "compression_flange_radius",
            "flange_centroid_distance",
            "second_moment_y",
            "second_moment_compression_flange",
            "torsional_constant",
        ],
    )
    if absent:
        result.add(
            LimitState("CFY", "Compression flange yielding", "F4-1", note=missing_note(absent))
        )
        return result

    S_xc = properties.section_modulus_compression
    S_xt = properties.section_modulus_tension
    root = sqrt(E / F_y)

    M_yc = F_y * S_xc  # Eq. F4-4
    M_yt = F_y * S_xt
    M_p = min(F_y * properties.plastic_modulus_x, 1.6 * F_y * properties.section_modulus_x)

    lam_w = properties.web_slenderness
    lam_pw = web.limiting_compact if web and web.known else 3.76 * root
    lam_rw = web.limiting_noncompact if web and web.known else 5.70 * root
    inertia_ratio = properties.second_moment_compression_flange / properties.second_moment_y

    R_pc = calculate_web_plastification_factor(M_p, M_yc, lam_w, lam_pw, lam_rw, inertia_ratio)
    R_pt = calculate_web_plastification_factor(M_p, M_yt, lam_w, lam_pw, lam_rw, inertia_ratio)
    F_L = calculate_limiting_flange_stress(F_y, S_xt, S_xc)
    # Eq. F4-5 user note: take J = 0 when I_yc/I_y <= 0.23
    J = properties.torsional_constant if inertia_ratio > 0.23 else 0.0

    L_p = calculate_limiting_yield_length(F_y, properties.compression_flange_radius, E)
    L_r = calculate_limiting_inelastic_length(
        F_L, properties.compression_flange_radius, J, S_xc, properties.flange_centroid_distance, E
    )
    result.intermediate_values.update(
        {
            "M_p": M_p,
            "M_yc": M_yc,
            "M_yt": M_yt,
            "R_pc": R_pc,
            "R_pt": R_pt,
            "F_L": F_L,
            "L_p": L_p,
            "L_r": L_r,
            "r_t": properties.compression_flange_radius,
            "J_effective": J,
        }
    )

    result.add(LimitState("CFY", "Compression flange yielding", "F4-1", R_pc * M_yc))

    if L_b <= L_p:
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling",
                "F4-2",
                note="L_b <= L_p — the limit state does not apply",
            )
        )
    elif L_b <= L_r:
        M_n = C_b * (R_pc * M_yc - (R_pc * M_yc - F_L * S_xc) * (L_b - L_p) / (L_r - L_p))
        result.add(
            LimitState(
                "LTB", "Lateral-torsional buckling (inelastic)", "F4-2", min(M_n, R_pc * M_yc)
            )
        )
    else:
        F_cr = calculate_elastic_lateral_torsional_buckling_stress(
            L_b,
            properties.compression_flange_radius,
            J,
            S_xc,
            properties.flange_centroid_distance,
            C_b,
            E,
        )
        result.intermediate_values["F_cr"] = F_cr
        result.add(
            LimitState(
                "LTB",
                "Lateral-torsional buckling (elastic)",
                "F4-3 / F4-5",
                min(F_cr * S_xc, R_pc * M_yc),
            )
        )

    _add_flange_local_buckling(result, properties, flange, R_pc * M_yc, F_L, S_xc, E)

    if S_xt >= S_xc:
        result.add(
            LimitState(
                "TFY",
                "Tension flange yielding",
                "F4-15",
                note="S_xt >= S_xc — the limit state does not apply",
            )
        )
    else:
        result.add(LimitState("TFY", "Tension flange yielding", "F4-15", R_pt * M_yt))
    return result


def _add_flange_local_buckling(
    result: FlexuralStrength,
    properties: SectionProperties,
    flange: Optional[SlendernessCheck],
    compression_flange_capacity: float,
    limiting_flange_stress: float,
    section_modulus_compression: float,
    youngs_modulus: float,
) -> None:
    """Append the F4.3 compression flange local buckling limit state."""
    if flange is None or not flange.known:
        result.add(
            LimitState(
                "FLB",
                "Compression flange local buckling",
                "F4-13",
                note=missing_note(["flange_slenderness"]),
            )
        )
        return
    if flange.classification is ElementClass.COMPACT:
        result.add(
            LimitState(
                "FLB",
                "Compression flange local buckling",
                "F4-13",
                note="compact flange — the limit state does not apply",
            )
        )
    elif flange.classification is ElementClass.NONCOMPACT:
        M_n = interpolate_noncompact(
            compression_flange_capacity,
            limiting_flange_stress * section_modulus_compression,
            flange.slenderness,
            flange.limiting_compact,
            flange.limiting_noncompact,
        )
        result.add(
            LimitState("FLB", "Compression flange local buckling (noncompact)", "F4-13", M_n)
        )
    else:
        k_c = properties.flange_buckling_coefficient or 0.76
        M_n = 0.9 * youngs_modulus * k_c * section_modulus_compression / flange.slenderness**2
        result.add(LimitState("FLB", "Compression flange local buckling (slender)", "F4-14", M_n))
