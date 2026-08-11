"""Modeling tools"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from Pynite import FEModel3D
from Pynite.LoadCombo import LoadCombo


def get_load_combos_from_tags(model: FEModel3D, tags=str | list[str]) -> dict[str, LoadCombo]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        for name, combo in model.load_combos.items():
            if tag in combo.combo_tags:
                filtered_combos[name] = combo

    return filtered_combos


def get_moments_from_tags(
    model: FEModel3D,
    member_name: str = "m_1",
    tags: str | list[str] = "strength",
    direction: Literal["My", "Mz"] = "Mz",
    n_points: int = 500,
) -> tuple[list, list[list], list, dict]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        filtered_combos.update(get_load_combos_from_tags(model, tag))

    moments_dict = {}
    member = model.members[member_name]
    for combo in filtered_combos.values():
        x, M = member.moment_array(direction, n_points, combo_name=combo.name)
        combo_str = (
            combo.name + ": " + "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
        )
        combo_str = combo_str.replace("+ -", "-")
        moments_dict[combo.name] = {"moments_list": M, "label": combo_str}

    moments = [item["moments_list"] for item in moments_dict.values()]
    labels = [item["label"] for item in moments_dict.values()]

    return x, moments, labels, moments_dict


def get_deflections_from_tags(
    model: FEModel3D,
    member_name: str = "m_1",
    tags: str | list[str] = "service",
    direction: Literal["dx", "dy", "dz"] = "dy",
    n_points: int = 500,
) -> tuple[list, list[list], list, dict]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        filtered_combos.update(get_load_combos_from_tags(model, tag))

    deflections_dict = {}
    member = model.members[member_name]
    for combo in filtered_combos.values():
        x, M = member.deflection_array(direction, n_points, combo_name=combo.name)
        combo_str = (
            combo.name + ": " + "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
        )
        combo_str = combo_str.replace("+ -", "-")
        deflections_dict[combo.name] = {"deflections_list": M, "label": combo_str}

    deflections = [item["deflections_list"] for item in deflections_dict.values()]
    labels = [item["label"] for item in deflections_dict.values()]

    return x, deflections, labels, deflections_dict


def get_shears_from_tags(
    model: FEModel3D,
    member_name: str = "m_1",
    tags: str | list[str] = "strength",
    direction: Literal["Fy", "Fz"] = "Fy",
    n_points: int = 500,
) -> tuple[list, list[list], list, dict]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        filtered_combos.update(get_load_combos_from_tags(model, tag))

    shears_dict = {}
    member = model.members[member_name]
    for combo in filtered_combos.values():
        x, M = member.shear_array(direction, n_points, combo_name=combo.name)
        combo_str = (
            combo.name + ": " + "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
        )
        combo_str = combo_str.replace("+ -", "-")
        shears_dict[combo.name] = {"shears_list": M, "label": combo_str}

    shears = [item["shears_list"] for item in shears_dict.values()]
    labels = [item["label"] for item in shears_dict.values()]

    return x, shears, labels, shears_dict


def get_axial_from_tags(
    model: FEModel3D,
    member_name: str = "m_1",
    tags: str | list[str] = "strength",
    n_points: int = 500,
) -> tuple[list, list[list], list, dict]:
    if isinstance(tags, str):
        tags = [tags]

    filtered_combos = {}
    for tag in tags:
        filtered_combos.update(get_load_combos_from_tags(model, tag))

    axial_dict = {}
    member = model.members[member_name]
    for combo in filtered_combos.values():
        x, M = member.axial_array(n_points, combo_name=combo.name)
        combo_str = (
            combo.name + ": " + "$" + " + ".join(f"{factor}{load}" for load, factor in combo.factors.items()) + "$"
        )
        combo_str = combo_str.replace("+ -", "-")
        axial_dict[combo.name] = {"axial_list": M, "label": combo_str}

    axial = [item["axial_list"] for item in axial_dict.values()]
    labels = [item["label"] for item in axial_dict.values()]

    return x, axial, labels, axial_dict


def get_node_results_table(
    node,
    results: list[str] | None = None,
    unit_label: str = "in",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build a dataframe of node displacements/reactions.

    Args:
        node: PyNite node object.
        results: Result types to include. Defaults to ["DX", "DY", "DZ"].

    Returns:
        Tuple containing:
            - DataFrame
            - Column mapping dictionary suitable for display_table
    """
    if results is None:
        results = [
            "DX",
            "DY",
            "DZ",
        ]

    data = {}

    for result in results:
        values = getattr(node, result)
        data[result] = {combo: float(value) for combo, value in values.items() if "all" not in combo}

    df = pd.DataFrame(data)
    df.index.name = "Load Combo"

    column_map = {
        "DX": rf"$\Delta_X$ [{unit_label}]",
        "DY": rf"$\Delta_Y$ [{unit_label}]",
        "DZ": rf"$\Delta_Z$ [{unit_label}]",
        "RX": rf"$\theta_X$ [{unit_label}]",
        "RY": rf"$\theta_Y$ [{unit_label}]",
        "RZ": rf"$\theta_Z$ [{unit_label}]",
        "RxnFX": rf"$R_X$ [{unit_label}]",
        "RxnFY": rf"$R_Y$ [{unit_label}]",
        "RxnFZ": rf"$R_Z$ [{unit_label}]",
        "RxnMX": rf"$M_X$ [{unit_label}]",
        "RxnMY": rf"$M_Y$ [{unit_label}]",
        "RxnMZ": rf"$M_Z$ [{unit_label}]",
    }

    return df, {column: column_map.get(column, column) for column in df.columns}


def get_node_results_for_combo_table(
    nodes: dict,
    combo_name: str,
    node_names: str | list[str] | None = None,
    results: list[str] | None = None,
    unit_label: str = "in",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Create a DataFrame of node results for a load combination.

    Args:
        nodes: PyNite node dictionary.
        combo_name: Load combination name.
        node_names: Nodes to include. If None, include all nodes.
        results: Result types to include. Defaults to ["DX", "DY", "DZ"].

    Returns:
        Tuple containing:
            - DataFrame indexed by node name.
            - column_names_filter_and_map dictionary suitable for
              display_table().
    """
    if results is None:
        results = ["DX", "DY", "DZ"]

    if isinstance(node_names, str):
        node_names = [node_names]

    if node_names is None:
        node_names = list(nodes.keys())

    rows = []

    for node_name in node_names:
        node = nodes[node_name]

        row = {}

        for result in results:
            result_dict = getattr(node, result, {})

            if combo_name in result_dict:
                row[result] = float(result_dict[combo_name])

        rows.append(pd.Series(row, name=node_name))

    df = pd.DataFrame(rows)
    df.index.name = "Node"

    column_map = {
        "DX": rf"$\Delta_x$ [{unit_label}]",
        "DY": rf"$\Delta_y$ [{unit_label}]",
        "DZ": rf"$\Delta_z$ [{unit_label}]",
        "RX": rf"$\theta_x$ [{unit_label}]",
        "RY": rf"$\theta_y$ [{unit_label}]",
        "RZ": rf"$\theta_z$ [{unit_label}]",
        "RxnFX": rf"$R_X$ [{unit_label}]",
        "RxnFY": rf"$R_Y$ [{unit_label}]",
        "RxnFZ": rf"$R_Z$ [{unit_label}]",
        "RxnMX": rf"$M_X$ [{unit_label}]",
        "RxnMY": rf"$M_Y$ [{unit_label}]",
        "RxnMZ": rf"$M_Z$ [{unit_label}]",
    }

    return df, {col: column_map.get(col, col) for col in df.columns}


def get_load_combos_table(
    load_combos: dict[str, LoadCombo],
    combo_names: str | list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Create a human-readable DataFrame of load combinations.

    Args:
        load_combos: PyNite load combination dictionary.
        combo_names: Load combinations to include. If None, include all.
        show_debug_combos: Choose to show debug combos. Overridden if 'combo_names' is given

    Returns:
        Tuple containing:
            DataFrame indexed by load combination name.
            Column mapping dictionary suitable for display_table().
    """
    if isinstance(combo_names, str):
        combo_names = [combo_names]

    if combo_names is None:
        filtered_combos = {}
        for name, combo in load_combos.items():
            if "all nominal" not in combo.combo_tags:
                filtered_combos[name] = combo
        combo_names = list(filtered_combos.keys())

    rows = []

    for combo_name in combo_names:
        combo = load_combos[combo_name]
        expression = _format_load_combo_expression(combo.factors)

        rows.append({
            "Load Combo": combo_name,
            "expression": expression,
        })

    dataframe = pd.DataFrame(rows).set_index("Load Combo")

    column_map = {
        "expression": "Expression",
    }

    return dataframe, column_map


def _format_load_combo_expression(factors: dict[str, float]) -> str:
    """Format a load combination factors dictionary as a readable expression.

    Args:
        factors: Dictionary mapping load case names to load factors.

    Returns:
        Human-readable load combination expression.
    """
    terms = []

    for index, (load_case, factor) in enumerate(factors.items()):
        factor = float(factor)

        if factor == 0:
            continue

        factor_text = f"{abs(factor):g}{load_case}"

        if index == 0:
            sign = "-" if factor < 0 else ""
        else:
            sign = " - " if factor < 0 else " + "

        terms.append(f"{sign}{factor_text}")
    terms.insert(0, "$")
    terms.append("$")
    return "".join(terms)
