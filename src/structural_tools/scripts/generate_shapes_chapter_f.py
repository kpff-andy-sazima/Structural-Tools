"""Generate shapes.js for the AISC 360-22 Chapter F flexural calculator.

Reads the AISC Shapes Database v16.0 through steelpy and emits a JavaScript
object containing every property the Chapter F engine needs.

Notes on the steelpy API (verified against steelpy 1.1.1):
    * Shape collections are stored in the ``aisc.profiles`` dict and are exposed
      only through ``Steel.__getattr__``. They do NOT appear in ``dir(aisc)``,
      so the collections must be read from ``aisc.profiles`` directly.
    * ``Profile.sections`` is a dict of designation -> Section.
    * ``Section.properties`` is a dict of property name -> value.
    * steelpy does not tabulate any width-to-thickness ratios, so every
      slenderness value required by Table B4.1b is computed here from the
      geometry. Formulas are verified against the printed Manual in the
      module docstring of :func:`slenderness`.
    * Some CSV cells hold numbers as strings and use an en dash for missing
      values, so all reads go through :func:`as_float`.

Usage:
    python generate_shapes_chapter_f.py                 # write ./shapes.js
    python generate_shapes_chapter_f.py -o web/shapes.js
    python generate_shapes_chapter_f.py --check         # verify against AISC
    python generate_shapes_chapter_f.py --inspect W     # dump available props
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# steelpy collection name -> calculator family code
# --------------------------------------------------------------------------- #
PROFILE_FAMILY: dict[str, str] = {
    "W_shapes": "W",
    "M_shapes": "M",
    "S_shapes": "S",
    "HP_shapes": "HP",
    "C_shapes": "C",
    "MC_shapes": "MC",
    "HSS_shapes": "HSS_RECT",
    "HSS_R_shapes": "HSS_ROUND",
    "PIPE_shapes": "PIPE",
    "WT_shapes": "WT",
    "MT_shapes": "WT",
    "ST_shapes": "WT",
    "L_shapes": "L",
    "DBL_L_shapes": "2L",
}

#: Properties copied straight across, per family. Left side is the calculator
#: key, right side is the steelpy property name.
DIRECT: dict[str, dict[str, str]] = {
    "I": {
        "A": "area", "d": "d", "bf": "bf", "tf": "tf", "tw": "tw", "kdes": "k",
        "Ix": "Ix", "Zx": "Zx", "Sx": "Sx", "rx": "rx",
        "Iy": "Iy", "Zy": "Zy", "Sy": "Sy", "ry": "ry",
        "J": "J", "Cw": "Cw", "rts": "rts", "ho": "ho",
    },
    "C": {
        "A": "area", "d": "d", "bf": "bf", "tf": "tf", "tw": "tw", "kdes": "k",
        "Ix": "Ix", "Zx": "Zx", "Sx": "Sx", "rx": "rx",
        "Iy": "Iy", "Zy": "Zy", "Sy": "Sy", "ry": "ry",
        "J": "J", "Cw": "Cw", "rts": "rts", "ho": "ho",
        "xbar": "x", "eo": "eo",
    },
    "HSS_RECT": {
        "A": "area", "Ht": "Ht", "B": "B", "tdes": "tdes",
        "Ix": "Ix", "Zx": "Zx", "Sx": "Sx", "rx": "rx",
        "Iy": "Iy", "Zy": "Zy", "Sy": "Sy", "ry": "ry", "J": "J",
    },
    "HSS_ROUND": {
        "A": "area", "OD": "OD", "tdes": "tdes",
        "Ix": "Ix", "Zx": "Zx", "Sx": "Sx", "rx": "rx", "J": "J",
    },
    "WT": {
        "A": "area", "d": "d", "bf": "bf", "tf": "tf", "tw": "tw",
        "Ix": "Ix", "Zx": "Zx", "Sx": "Sx", "rx": "rx",
        "Iy": "Iy", "Zy": "Zy", "Sy": "Sy", "ry": "ry",
        "J": "J", "Cw": "Cw", "ybar": "y",
    },
    "2L": {
        "A": "area", "d": "d", "leg_b": "b", "t": "t",
        "Ix": "Ix", "Zx": "Zx", "Sx": "Sx", "rx": "rx",
        "Iy": "Iy", "Zy": "Zy", "Sy": "Sy", "ry": "ry", "ybar": "y",
    },
    "L": {
        "A": "area", "leg_d": "d", "leg_b": "b", "t": "t",
        "Ix": "Ix", "Zx": "Zx", "Sx": "Sx", "rx": "rx",
        "Iy": "Iy", "Zy": "Zy", "Sy": "Sy", "ry": "ry",
        "rz": "rz", "Sz": "Sz", "J": "J", "xbar": "x", "ybar": "y",
    },
}

FAMILY_GROUP: dict[str, str] = {
    "W": "I", "M": "I", "S": "I", "HP": "I",
    "C": "C", "MC": "C",
    "HSS_RECT": "HSS_RECT", "HSS_ROUND": "HSS_ROUND", "PIPE": "HSS_ROUND",
    "WT": "WT", "2L": "2L", "L": "L",
}


def as_float(value: Any) -> float | None:
    """Coerce a steelpy property to a float.

    steelpy stores some CSV cells as strings and marks missing values with an
    en dash, so a plain ``float()`` is not safe.

    Args:
        value: Raw value from a Section's property dict.
    Returns:
        The numeric value, or None when the cell is blank or non-numeric.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip().replace(",", "")
        if text in {"", "-", "\u2013", "\u2014", "N/A", "NA", "nan"}:
            return None
        try:
            result = float(text)
        except ValueError:
            return None
    if result != result:  # NaN
        return None
    return round(result, 6)


def get(section: Any, name: str) -> float | None:
    """Read one numeric property off a steelpy Section.

    Args:
        section: steelpy Section object.
        name: steelpy property name.
    Returns:
        The numeric value, or None when absent.
    """
    return as_float(section.properties.get(name))


def slenderness(group: str, rec: dict[str, Any], section: Any) -> None:
    """Compute the Table B4.1b width-to-thickness ratios in place.

    steelpy tabulates no slenderness ratios, so they are derived here:

    * I-shapes and tees, flange (case 10): ``bf / (2 tf)``
    * Channel flange (case 10): ``bf / tf`` — b is the full flange width
    * I-shape and channel web (case 15): ``h / tw`` with ``h = d - 2k``
    * Tee stem (case 14): ``d / tw``
    * Rectangular HSS (cases 17, 19): ``b / tdes`` and ``h / tdes``, where b and
      h are the flat widths already tabulated by steelpy
    * Round HSS and pipe (case 20): ``OD / tdes``
    * Angle legs (case 12): ``b / t``

    Args:
        group: Coarse family group.
        rec: Record being built; mutated in place.
        section: Source steelpy Section.
    """
    if group in ("I", "C"):
        d, k, tw = rec.get("d"), rec.get("kdes"), rec.get("tw")
        bf, tf = rec.get("bf"), rec.get("tf")
        if None not in (d, k, tw) and tw:
            rec["h"] = round(d - 2 * k, 4)
            rec["h_tw"] = round(rec["h"] / tw, 3)
        if None not in (bf, tf) and tf:
            # Channels use the full flange width; I-shapes use half.
            rec["b_t" if group == "C" else "bf_2tf"] = round(
                bf / tf if group == "C" else bf / (2 * tf), 3
            )

    elif group == "WT":
        bf, tf, d, tw = rec.get("bf"), rec.get("tf"), rec.get("d"), rec.get("tw")
        if None not in (bf, tf) and tf:
            rec["bf_2tf"] = round(bf / (2 * tf), 3)
        if None not in (d, tw) and tw:
            rec["d_t"] = round(d / tw, 3)

    elif group == "HSS_RECT":
        # steelpy tabulates the flat widths b and h alongside the outside
        # dimensions B and Ht.
        flat_b, flat_h = get(section, "b"), get(section, "h")
        t = rec.get("tdes")
        if t:
            if flat_b:
                rec["b_t"] = round(flat_b / t, 3)
            if flat_h:
                rec["h_t"] = round(flat_h / t, 3)

    elif group == "HSS_ROUND":
        od, t = rec.get("OD"), rec.get("tdes")
        if od and t:
            rec["D_t"] = round(od / t, 3)

    elif group in ("L", "2L"):
        b, t, d = rec.get("leg_b"), rec.get("t"), rec.get("d")
        if b and t:
            rec["b_t"] = round(b / t, 3)
        if d and t:
            rec["d_t"] = round(d / t, 3)


def single_angle_key(double_name: str, singles: dict[str, Any]) -> str | None:
    """Find the single-angle designation underlying a double-angle name.

    Double-angle designations optionally carry a back-to-back separation token
    and an unequal-leg orientation suffix, always in that order::

        DBL_L4X4X1_4            -> L4X4X1_4
        DBL_L4X4X1_4X3_8        -> L4X4X1_4   (3/8 in. separation)
        DBL_L8X6X1LLBB          -> L8X6X1     (long legs back to back)
        DBL_L8X6X1X3_8LLBB      -> L8X6X1     (both)

    The orientation suffix is stripped first, then trailing ``X``-delimited
    separation tokens are peeled off until a real single angle is matched.

    Args:
        double_name: steelpy DBL_L designation.
        singles: Mapping of single-angle designation to Section.
    Returns:
        The matching single-angle designation, or None when unmatched.
    """
    if not double_name.startswith("DBL_"):
        return None
    stem = re.sub(r"(LLBB|SLBB)$", "", double_name[len("DBL_"):])
    tokens = stem.split("X")
    for stop in range(len(tokens), 1, -1):
        candidate = "X".join(tokens[:stop])
        if candidate in singles:
            return candidate
    return None


def double_angle_j(double_name: str, singles: dict[str, Any]) -> float | None:
    """Compute J for a double angle as twice the single-angle value.

    steelpy's DBL_L table omits J, which Section F9 needs for ``Mcr``.

    Args:
        double_name: steelpy DBL_L designation.
        singles: Mapping of single-angle designation to Section.
    Returns:
        Twice the single-angle torsional constant, or None when unmatched.
    """
    key = single_angle_key(double_name, singles)
    if key is None:
        return None
    j_single = get(singles[key], "J")
    return None if j_single is None else round(2 * j_single, 6)


def js_key(name: str) -> str:
    """Convert an AISC designation into a valid JavaScript identifier.

    Args:
        name: AISC shape designation, e.g. "W6X8.5" or "HSS6X6X3/8".
    Returns:
        Sanitized key, e.g. "W6X8_5".
    """
    key = re.sub(r"[^0-9A-Za-z_]", "_", name.upper())
    return key if key[0].isalpha() or key[0] == "_" else f"_{key}"


def load_aisc() -> Any:
    """Import the steelpy database.

    Returns:
        The steelpy ``aisc`` root object.
    """
    try:
        from steelpy import aisc  # type: ignore
    except ImportError:
        sys.exit("steelpy is not installed — pip install steelpy")
    return aisc


def build() -> dict[str, dict[str, Any]]:
    """Extract every shape from steelpy into calculator-ready dictionaries.

    Returns:
        Mapping of JavaScript key to the shape's property dictionary.
    """
    aisc = load_aisc()
    profiles = aisc.profiles
    singles = profiles["L_shapes"].sections if "L_shapes" in profiles else {}

    out: dict[str, dict[str, Any]] = {}
    for profile_name, family in PROFILE_FAMILY.items():
        collection = profiles.get(profile_name)
        if collection is None:
            print(f"  ! collection {profile_name} not found — skipped")
            continue

        group = FAMILY_GROUP[family]
        mapping = DIRECT[group]
        count = 0

        for designation, section in collection.sections.items():
            rec: dict[str, Any] = {"family": family}
            for target, source in mapping.items():
                value = get(section, source)
                if value is not None:
                    rec[target] = value

            slenderness(group, rec, section)

            # --- family-specific repairs ------------------------------------
            if family == "2L":
                j = double_angle_j(designation, singles)
                if j is not None:
                    rec["J"] = j
            if group == "L":
                # Equal-leg angles have betaw = 0; unequal legs must be entered
                # by hand because the Manual tabulates betaw separately.
                if rec.get("leg_b") == rec.get("leg_d"):
                    rec["betaw"] = 0.0
                sw = get(section, "SwA") or get(section, "SwC")
                if sw:
                    rec["Sw"] = sw
                rec["Sc"] = rec.get("Sx")
            if group == "HSS_ROUND" and "J" not in rec and "Ix" in rec:
                rec["J"] = round(2 * rec["Ix"], 6)

            if len(rec) > 1:
                out[js_key(designation)] = rec
                count += 1

        print(f"  {profile_name:16s} -> {family:9s} {count:4d} shapes")
    return out


def build_quiet() -> dict[str, dict[str, Any]]:
    """Run :func:`build` with its progress output suppressed.

    Returns:
        Mapping of JavaScript key to the shape's property dictionary.
    """
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return build()


def emit(data: dict[str, dict[str, Any]], path: Path) -> None:
    """Write the JavaScript shape database.

    Args:
        data: Mapping of JavaScript key to property dictionary.
        path: Destination file.
    """
    families = sorted({v["family"] for v in data.values()})
    lines = [
        "// AUTO-GENERATED by generate_shapes_chapter_f.py from steelpy — do not edit.",
        "// AISC Shapes Database v16.0 (kip-inch).",
        "// Width-to-thickness ratios are computed by the generator; r_ts, c, r_t,",
        "// Sxc/Sxt, Rpc, Rpg and kc are derived in-browser by the calculator.",
        f"// {len(data)} shapes across {len(families)} families: {', '.join(families)}",
        "const SHAPES = {",
    ]
    for key in sorted(data):
        body = ", ".join(f"{k}: {json.dumps(v)}" for k, v in data[key].items())
        lines.append(f"  {key}: {{ {body} }},")
    lines.append("};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {path} — {len(data)} shapes, {path.stat().st_size / 1024:.0f} kB")


#: (shape, key, expected, tolerance) sampled from the AISC Manual, 16th Ed.
#: The h/tw entries use h = d - 2*kdes per Section B4.1b, which is the
#: fillet-to-fillet clear web depth. Do not substitute the tabulated T
#: dimension: T is based on the detailing k and gives a different ratio
#: (W18X50 -> 43.7 from T versus the published 45.2 from kdes).
CHECKS: list[tuple[str, str, float, float]] = [
    ("W18X50", "bf_2tf", 6.57, 0.02),
    ("W18X50", "h_tw", 45.2, 0.1),
    ("W18X50", "Zx", 101.0, 0.5),
    ("W18X50", "Sx", 88.9, 0.5),
    ("W18X50", "rts", 1.98, 0.01),
    ("W18X50", "ho", 17.4, 0.05),
    ("W21X48", "bf_2tf", 9.47, 0.02),
    ("W14X90", "bf_2tf", 10.2, 0.05),
    ("W14X90", "Zx", 157.0, 0.5),
    ("W44X230", "h_tw", 54.8, 0.2),
    ("C15X33_9", "b_t", 5.23, 0.02),
    ("C15X33_9", "h_tw", 30.3, 0.2),
    ("C15X33_9", "Zx", 50.8, 0.3),
    ("HSS8X8X1_2", "b_t", 14.2, 0.05),
    ("HSS8X8X1_2", "Zx", 37.5, 0.2),
    ("HSS12X8X1_4", "h_t", 48.5, 0.2),
    ("HSS10_000X0_375", "D_t", 28.7, 0.1),
    ("PIPE5STD", "D_t", 23.1, 0.2),
    ("WT9X25", "d_t", 25.4, 0.1),
    ("L4X4X1_4", "b_t", 16.0, 0.05),
    ("DBL_L4X4X1_4", "b_t", 16.0, 0.05),
]


def check(data: dict[str, dict[str, Any]]) -> int:
    """Compare generated values against published AISC values.

    Args:
        data: Generated shape database.
    Returns:
        Process exit code: 0 when every check passes, 1 otherwise.
    """
    failures = 0
    print("\nverification against AISC Manual, 16th Ed.:")
    for name, key, expected, tol in CHECKS:
        actual = data.get(name, {}).get(key)
        if actual is None:
            print(f"  FAIL {name:18s} {key:8s} missing")
            failures += 1
        elif abs(actual - expected) > tol:
            print(f"  FAIL {name:18s} {key:8s} {actual} != {expected}")
            failures += 1
        else:
            print(f"  ok   {name:18s} {key:8s} {actual:>8} (AISC {expected})")

    # Structural completeness: every shape must carry the slenderness ratios
    # its Chapter F route depends on, or it will silently classify as compact.
    required = {
        "I": ("bf_2tf", "h_tw", "Zx", "Sx", "ry", "J", "ho"),
        "C": ("b_t", "h_tw", "Zx", "Sx", "ry", "J", "ho"),
        "HSS_RECT": ("b_t", "h_t", "Zx", "Sx", "ry", "J"),
        "HSS_ROUND": ("D_t", "Zx", "Sx"),
        "WT": ("bf_2tf", "d_t", "Zx", "Iy", "J", "ry"),
        "2L": ("b_t", "Zx", "Iy", "ry", "J"),
        "L": ("b_t", "Sx", "rz"),
    }
    for family, keys in required.items():
        members = {n: v for n, v in data.items()
                   if FAMILY_GROUP.get(v["family"]) == family}
        for key in keys:
            gaps = [n for n, v in members.items() if key not in v]
            if gaps:
                print(f"  WARN {len(gaps):4d}/{len(members)} {family} shapes "
                      f"missing {key!r}, e.g. {gaps[:3]}")
                failures += 1

    print(f"\n{'all checks passed' if not failures else f'{failures} check(s) FAILED'}")
    return 1 if failures else 0


#: Shapes embedded in the calculator as an offline fallback when shapes.js is
#: absent. Chosen to exercise every Chapter F route at least once.
DEMO_SHAPES: tuple[str, ...] = (
    "W6X15", "W12X26", "W14X90", "W16X26", "W18X50", "W21X48", "W24X68",
    "C12X20_7", "C15X33_9",
    "HSS6X6X3_8", "HSS8X8X1_2", "HSS12X8X1_4",
    "HSS6_000X0_375", "HSS10_000X0_375", "PIPE5STD",
    "WT9X25", "WT6X20", "L4X4X1_4", "L6X4X1_2", "DBL_L4X4X1_4",
)


def emit_demo(data: dict[str, dict[str, Any]]) -> None:
    """Print a DEMO_SHAPES block for pasting into the calculator HTML.

    The calculator ships a small offline fallback set. Hand-typing those
    numbers is error prone, so this regenerates the block from steelpy.

    Args:
        data: Generated shape database.
    """
    print("      const DEMO_SHAPES = {")
    for name in DEMO_SHAPES:
        rec = data.get(name)
        if rec is None:
            print(f"        // {name} not found in the database")
            continue
        body = ", ".join(f"{k}: {json.dumps(v)}" for k, v in rec.items())
        print(f"        {name}: {{ {body} }},")
    print("      };")


def inspect(prefix: str) -> None:
    """Print the properties steelpy exposes for the first matching shape.

    Args:
        prefix: Designation prefix to look for, e.g. "W" or "HSS".
    """
    aisc = load_aisc()
    for profile_name, collection in aisc.profiles.items():
        for designation, section in collection.sections.items():
            if designation.upper().startswith(prefix.upper()):
                print(f"{profile_name} / {designation}")
                for key, value in section.properties.items():
                    print(f"  {key:8s} = {value!r}")
                return
    print(f"no shape found starting with {prefix!r}")


def main() -> None:
    """Parse arguments and run the generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default="shapes.js", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="print the DEMO_SHAPES fallback block and exit")
    parser.add_argument("--inspect", metavar="PREFIX")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    if args.demo:
        emit_demo(build_quiet())
        return

    print("reading steelpy collections:")
    data = build()
    if not data:
        sys.exit("no shapes extracted — run with --inspect W to inspect the API")
    emit(data, args.out)
    if args.check:
        sys.exit(check(data))


if __name__ == "__main__":
    main()
