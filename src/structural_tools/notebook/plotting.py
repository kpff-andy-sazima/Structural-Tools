"""Plot functions"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from enum import Enum
from math import isclose

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from Pynite import FEModel3D, PhysMember, Node3D

from typing import Any

RCPARAMS_DICT: dict = {
    "figure.figsize": (12, 3),
    "figure.dpi": 250,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.grid": True,
    "axes.axisbelow": True,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "DejaVu Sans"],
    # "text.usetex": True,
}

LOAD_COLORS = {
    "D": "#d62728",
    "L": "#1f77b4",
    "Lr": "#17becf",
    "S": "#9467bd",
    "R": "#ff7f0e",
    "W": "#2ca02c",
}


class UpAxis(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"


def set_plot_style(rcParams_dict: dict | None = None, append_to_structural_tools_defaults: bool = True) -> None:
    if rcParams_dict:
        if append_to_structural_tools_defaults:
            plt.rcParams.update(RCPARAMS_DICT)

        plt.rcParams.update(rcParams_dict)
    else:
        plt.rcParams.update(RCPARAMS_DICT)


def plot_internal_diagram(
    x: np.ndarray,
    ys: list[np.ndarray] | np.ndarray,
    labels: list[str] | str,
    xlabel: str = "Location",
    ylabel: str = "Value",
    title: str = "Diagram",
    figsize: tuple[float, float] = (12, 3),
    envelope: bool = False,
    show_member: bool = True,
    show_plot: bool = True,
    save_png: str | None = None,
) -> tuple[Figure, Axes]:

    if not show_plot:
        plt.ioff

    fig, ax = plt.subplots(figsize=figsize)

    if not isinstance(ys, list):
        ys = [ys]
    if not isinstance(labels, list):
        labels = [labels]

    assert len(ys) == len(labels)

    for i, y_value_set in enumerate(ys):
        ax.plot(x, y_value_set, label=labels[i])

    if envelope:
        y_values = []
        for line in ax.lines:
            y_value = np.array(line.get_ydata())
            y_values.append(y_value)

        y_values = np.vstack(y_values)
        env_max = np.max(y_values, axis=0)
        env_min = np.min(y_values, axis=0)
        ax.fill_between(x, env_min, env_max, color="gray", alpha=0.25, zorder=1, label="_nolegend_")

    if show_member:
        ax.plot(
            [x[0], x[-1]],
            [0, 0],
            color="black",
            linewidth=3,
            solid_capstyle="butt",
            zorder=0,
            label="_nolegend_",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.legend()
    plt.tight_layout()
    if save_png:
        plt.savefig(save_png)
    if show_plot:
        plt.ion()
        plt.show()
    else:
        plt.close()
        plt.ion()
    return fig, ax


def _halo(lw: float = 2.5):
    """Return a white outline path effect for legible text over fills.

    Args:
        lw: Halo stroke width in points.
    Returns:
        List with a single Stroke path effect.
    """
    return [pe.withStroke(linewidth=lw, foreground="white")]


def _sig3(v: float) -> str:
    """Format a value to 3 significant figures.

    Args:
        v: Value to format.
    Returns:
        String with 3 significant figures.
    """
    return f"{v:.3g}"


def _v_arrow(ax, x, y_base, y_top, sign, color, lw=1.4, head=9):
    """Draw a vertical load arrow with a display-space fixed head.

    Args:
        ax: Target axes.
        x: X location.
        y_base: Beam-side y (bottom of the graphic).
        y_top: Far y (top of the graphic).
        sign: Negative points the tip toward the beam, positive away.
        color: Arrow color.
        lw: Shaft line width.
        head: Arrowhead size via mutation_scale (points), constant so heads
            never balloon on short shafts or shrink on long ones.
    """
    if sign < 0:
        y_start, y_end = y_top, y_base
    else:
        y_start, y_end = y_base, y_top
    ax.annotate(
        "",
        xy=(x, y_end),
        xytext=(x, y_start),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=head, shrinkA=0, shrinkB=0),
    )


def _member_nodes(member, L):
    """Return ordered (x, node) pairs along the member.

    Walks sub_members (PhysMember) so interior nodes are captured at their
    true local x. Falls back to the i/j end nodes when the member has not
    been descritized.

    Args:
        member: PyNite PhysMember or Member3D.
        L: Member length in local coordinates.
    Returns:
        List of (x, node) tuples ordered from the i-end to the j-end.
    """
    subs = getattr(member, "sub_members", None)
    if not subs:
        return [(0.0, member.i_node), (L, member.j_node)]
    nodes = []
    x_o = 0.0
    for i, sub in enumerate(subs.values()):
        if i == 0:
            nodes.append((x_o, sub.i_node))
        sub_L = sub.L() if callable(getattr(sub, "L")) else sub.L
        x_o += sub_L
        nodes.append((x_o, sub.j_node))
    return nodes


def _stagger_labels(xs, min_gap):
    """Assign label tiers so close-together x positions don't overlap.

    Args:
        xs: Sorted list of x positions.
        min_gap: Minimum x separation before bumping to a lower tier.
    Returns:
        List of integer tiers (0, 1, 2, ...) parallel to xs.
    """
    tiers = []
    last_x_by_tier = {}
    for x in xs:
        tier = 0
        while tier in last_x_by_tier and x - last_x_by_tier[tier] < min_gap:
            tier += 1
        last_x_by_tier[tier] = x
        tiers.append(tier)
    return tiers


def _plane_dofs(direction_filter):
    """Return the in-plane DOF names for a load direction filter.

    The diagram plane is defined by the transverse direction being plotted.
    The relevant DOFs are the axial translation, the transverse translation,
    and the in-plane rotation:
        FY -> ("DX", "DY", "RZ")   bending about local z
        FZ -> ("DX", "DZ", "RY")   bending about local y
    Any other or None filter defaults to the FY plane.

    Args:
        direction_filter: The load direction filter (e.g. "FY", "FZ").
    Returns:
        Tuple of (axial_dof, transverse_dof, rotation_dof) names.
    """
    if direction_filter is not None and direction_filter.upper() == "FZ":
        return ("DX", "DZ", "RY")
    return ("DX", "DY", "RZ")


def _classify_support(node, dofs=("DX", "DY", "RZ")):
    """Classify a node's restraint as fixed, pin, roller, or free.

    Reads PyNite ``support_*`` boolean flags but considers only the three
    in-plane DOFs of the diagram plane. Out-of-plane restraints are ignored
    so they can't over-classify a support. In-plane rule: both translations
    plus rotation reads as fixed; both translations with free rotation as
    pin; a single translation as roller.

    Args:
        node: PyNite Node3D (or any object exposing support_* booleans).
        dofs: (axial, transverse, rotation) support flag names from _plane_dofs.
    Returns:
        One of "fixed", "pin", "roller", or None when unrestrained.
    """
    axial_dof, trans_dof, rot_dof = dofs
    t_axial = bool(getattr(node, f"support_{axial_dof}", False))
    t_trans = bool(getattr(node, f"support_{trans_dof}", False))
    r_plane = bool(getattr(node, f"support_{rot_dof}", False))
    n_trans = t_axial + t_trans
    if n_trans == 0 and not r_plane:
        return None
    if n_trans >= 2 and r_plane:
        return "fixed"
    if n_trans >= 2:
        return "pin"
    return "roller"


def _draw_support(ax, x, kind, L, size, color="#333333"):
    """Draw a schematic support glyph hanging below the baseline at x.

    Pin is a gray-filled triangle, roller a gray triangle riding on rollers,
    fixed a hatched ground line.

    Args:
        ax: Target axes.
        x: X location of the support.
        kind: "fixed", "pin", or "roller".
        L: Member length, used to scale glyph width.
        size: Triangle/glyph height in y-data units.
        color: Glyph edge color.
    Returns:
        The lowest y-value reached by the glyph.
    """
    w = 0.012 * L
    fill = "0.7"
    if kind == "pin":
        ax.fill([x, x - w, x + w], [0, -size, -size], facecolor=fill, edgecolor=color, lw=1.5, zorder=6, clip_on=False)
        return -size
    if kind == "roller":
        h = 0.78 * size
        ax.fill([x, x - w, x + w], [0, -h, -h], facecolor=fill, edgecolor=color, lw=1.5, zorder=6, clip_on=False)
        for cx in (x - 0.5 * w, x + 0.5 * w):
            ax.plot([cx], [-h - 0.10 * size], "o", color=color, markersize=4, zorder=6, clip_on=False)
        ax.plot([x - w, x + w], [-size, -size], color=color, lw=1.2, zorder=6, clip_on=False)
        return -size
    # fixed: hatched ground line through the node
    bw = 1.4 * w
    n_hatch = 5
    ax.plot([x - bw, x + bw], [0, 0], color=color, lw=2.0, zorder=6, clip_on=False)
    for k in range(n_hatch):
        hx = x - bw + 2 * bw * k / (n_hatch - 1)
        ax.plot([hx, hx - 0.5 * bw], [0, -0.7 * size], color=color, lw=1.0, zorder=6, clip_on=False)
    return -0.7 * size


def plot_member_loads(
    member,
    direction_filter: str | None = "FY",
    include_node_loads: bool = True,
    color_by: str = "case",
    force_unit: str | None = "kip",
    length_unit: str | None = "inch",
    sort_by_case: bool = True,
    lane_height: float = 0.6,
    lane_pitch: float = 1.45,
    min_height_frac: float = 0.6,
    min_point_frac: float = 1.0,
    max_height_frac: float = 1.25,
    arrow_head: float = 9.0,
    figsize: tuple[float, float] = (12, 8),
    ax=None,
    save_png: str | None = None,
    show_plt: bool = False,
    show_nodes: bool = True,
    show_supports: bool = True,
    support_size: float = 0.55,
):
    """Plot a stacked loading diagram for one PyNite member.

    Reads the member's (and optionally every node along it) ``_load_metadata``
    populated by the add_named_* helpers. Each load occupies its own lane
    stacked above the member baseline at its correct x-location. Distributed
    loads render as trapezoids, point loads as single arrows with a leader
    line down to the beam. Point and distributed magnitudes are scaled
    independently. Supports are drawn from node restraint flags matched to
    the diagram plane. Load names are bold; values/units are normal weight.

    Args:
        member: PyNite PhysMember/Member3D whose loads to plot.
        direction_filter: Only include loads whose direction matches this,
            case-insensitive. Use None to include every direction.
        include_node_loads: Include loads on nodes along the member.
        color_by: Metadata key used to color loads ("case" or "category").
        force_unit: Force unit label. Defaults to "kip"; pass None to omit.
        length_unit: Length unit label. Defaults to "inch"; pass None to omit.
        sort_by_case: Group lanes by their color_by key (all of one case
            together) instead of insertion order.
        lane_height: Nominal lane graphic height in data units.
        lane_pitch: Vertical spacing between lane baselines.
        min_height_frac: Minimum dist graphic height as a fraction of lane_height.
        min_point_frac: Minimum point-load arrow length as a fraction of
            lane_height.
        max_height_frac: Maximum graphic height as a fraction of lane_height.
        arrow_head: Arrowhead size (points) passed to mutation_scale.
        figsize: Figure size when ax is None.
        ax: Existing axes; created if None.
        save_png: File path to save a PNG. None skips saving.
        show_plt: Call plt.show() before returning when True.
        show_nodes: Draw nodes as labeled dots on the baseline.
        show_supports: Draw fix/pin/roller glyphs from node support flags.
        support_size: Support glyph height in y-data units.
    Returns:
        The matplotlib Axes containing the diagram.
    """
    member_name = member.name
    L = member.L() if callable(getattr(member, "L")) else member.L
    meta = getattr(member, "_load_metadata", defaultdict(list))
    nodes_on_member = _member_nodes(member, L)

    def _keep(direction):
        if direction_filter is None:
            return True
        return direction.upper() == direction_filter.upper()

    def _fmt_dist(v):
        s = _sig3(v)
        if force_unit and length_unit:
            return f"{s} {force_unit}/{length_unit}"
        if force_unit:
            return f"{s} {force_unit}/len"
        return s

    def _fmt_pt(v):
        s = _sig3(v)
        return f"{s} {force_unit}" if force_unit else s

    lanes = []
    for d in meta.get("dist_loads", []):
        if not _keep(d["direction"]):
            continue
        lanes.append({
            "kind": "dist",
            "x1": 0.0 if d["x1"] is None else d["x1"],
            "x2": L if d["x2"] is None else d["x2"],
            "w1": d["w1"],
            "w2": d["w2"],
            "name": d["name"],
            "key": d.get(color_by) or d["case"],
        })
    for p in meta.get("point_loads", []):
        if not _keep(p["direction"]):
            continue
        lanes.append({
            "kind": "point",
            "x": p["x"],
            "P": p["P"],
            "name": p["name"],
            "key": p.get(color_by) or p["case"],
        })
    if include_node_loads:
        for x_pos, node in nodes_on_member:
            nmeta = getattr(node, "_load_metadata", None)
            if nmeta and nmeta.get("node_loads"):
                for nl in nmeta["node_loads"]:
                    if not _keep(nl["direction"]):
                        continue
                    lanes.append({
                        "kind": "point",
                        "x": x_pos,
                        "P": nl["P"],
                        "name": nl["name"],
                        "key": nl.get(color_by) or nl["case"],
                    })
            else:
                for direction, P, case, *_ in getattr(node, "NodeLoads", []):
                    if not _keep(direction):
                        continue
                    lanes.append({
                        "kind": "point",
                        "x": x_pos,
                        "P": P,
                        "name": case,
                        "key": case,
                    })

    # Group lanes by case (color_by key), preserving first-appearance order.
    key_order = []
    for ln in lanes:
        if ln["key"] not in key_order:
            key_order.append(ln["key"])
    if sort_by_case:
        order_idx = {k: i for i, k in enumerate(key_order)}
        lanes.sort(key=lambda ln: order_idx[ln["key"]])

    dist_mags = [abs(v) for ln in lanes if ln["kind"] == "dist" for v in (ln["w1"], ln["w2"])]
    pt_mags = [abs(ln["P"]) for ln in lanes if ln["kind"] == "point"]
    max_dist = max(dist_mags) if dist_mags else 1.0
    max_pt = max(pt_mags) if pt_mags else 1.0
    min_h = min_height_frac * lane_height
    min_pt_h = min_point_frac * lane_height
    max_h = max_height_frac * lane_height

    def _height(mag, scale, floor):
        h = lane_height * abs(mag) / scale
        return max(floor, min(max_h, h))

    cmap = plt.get_cmap("tab10")
    color_of = {k: cmap(i % 10) for i, k in enumerate(key_order)}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.plot([0, L], [0, 0], color="black", lw=3, zorder=5)

    for i, ln in enumerate(lanes):
        y0 = 0.5 + i * lane_pitch
        color = color_of[ln["key"]]
        if ln["kind"] == "dist":
            h1 = _height(ln["w1"], max_dist, min_h)
            h2 = _height(ln["w2"], max_dist, min_h)
            sign = -1 if (ln["w1"] + ln["w2"]) < 0 else 1
            ax.fill(
                [ln["x1"], ln["x2"], ln["x2"], ln["x1"]],
                [y0, y0, y0 + h2, y0 + h1],
                color=color,
                alpha=0.15,
                zorder=3,
            )
            ax.plot([ln["x1"], ln["x2"]], [y0 + h1, y0 + h2], color=color, lw=1.5, zorder=4)
            n = max(2, int((ln["x2"] - ln["x1"]) / max(L / 12, 1e-9)))
            for j in range(n + 1):
                t = j / n
                xa = ln["x1"] + (ln["x2"] - ln["x1"]) * t
                ha = h1 + (h2 - h1) * t
                _v_arrow(ax, xa, y0, y0 + ha, sign, color, head=arrow_head)
            # Uniform loads print one centered value; only trapezoidal loads
            # get the left/right corner pair. Values are normal weight.
            if isclose(ln["w1"], ln["w2"], rel_tol=1e-6, abs_tol=1e-12):
                xc = 0.5 * (ln["x1"] + ln["x2"])
                ax.annotate(
                    _fmt_dist(ln["w1"]),
                    xy=(xc, y0 + max(h1, h2)),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                    path_effects=_halo(),
                )
            else:
                ax.annotate(
                    _fmt_dist(ln["w1"]),
                    xy=(ln["x1"], y0 + h1),
                    xytext=(2, 3),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color=color,
                    path_effects=_halo(),
                )
                ax.annotate(
                    _fmt_dist(ln["w2"]),
                    xy=(ln["x2"], y0 + h2),
                    xytext=(-2, 3),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color=color,
                    path_effects=_halo(),
                )
            ax.annotate(
                ln["name"],
                xy=(ln["x2"], y0 + max(h1, h2) / 2),
                xytext=(12, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
                clip_on=False,
            )
        else:
            h = _height(ln["P"], max_pt, min_pt_h)
            sign = -1 if ln["P"] < 0 else 1
            # Leader line ties the elevated arrow back to its x on the beam.
            ax.plot([ln["x"], ln["x"]], [0, y0], color=color, lw=0.7, ls=":", alpha=0.55, zorder=2)
            _v_arrow(ax, ln["x"], y0, y0 + h, sign, color, lw=2.2, head=arrow_head + 2)
            ax.annotate(
                _fmt_pt(ln["P"]),
                xy=(ln["x"], y0 + h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                path_effects=_halo(),
            )
            ax.annotate(
                ln["name"],
                xy=(ln["x"], y0 + h / 2),
                xytext=(12, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
                clip_on=False,
            )

    n_dots = 0
    label_bottom = 0.0
    if show_nodes:
        pairs = sorted(nodes_on_member, key=lambda p: p[0])
        node_xs = [x for x, _ in pairs]
        ax.plot(node_xs, [0] * len(node_xs), "s", color="black", markersize=8, zorder=7, clip_on=False)

        sup_reach = 0.0
        if show_supports:
            plane_dofs = _plane_dofs(direction_filter)
            for x, node in pairs:
                kind = _classify_support(node, plane_dofs)
                if kind is None:
                    continue
                low = _draw_support(ax, x, kind, L, support_size)
                sup_reach = min(sup_reach, low)

        label_top = sup_reach - 0.18
        step = 0.34
        tiers = _stagger_labels(node_xs, min_gap=0.07 * L)
        n_dots = (max(tiers) + 1) if tiers else 0
        for (x, node), tier in zip(pairs, tiers):
            ax.text(
                x,
                label_top - tier * step,
                node.name,
                ha="center",
                va="top",
                fontsize=8,
                color="black",
                clip_on=False,
                path_effects=_halo(),
            )
        label_bottom = label_top - max(0, n_dots - 1) * step

    top = 0.5 + len(lanes) * lane_pitch + 0.5 if lanes else 1.0
    bottom = (label_bottom - 0.45) if show_nodes else -0.5
    ax.set_xlim(-0.05 * L, 1.05 * L)
    ax.set_ylim(bottom, top)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=6)

    # Gridlines clipped to the load region (beam up), not through supports.
    ax.grid(False)
    for xt in ax.get_xticks():
        if -0.05 * L <= xt <= 1.05 * L:
            ax.plot([xt, xt], [0, top], color="0.88", lw=0.8, zorder=0)

    xlabel = "Distance along member"
    if length_unit:
        xlabel += f" ({length_unit})"
    ax.set_xlabel(xlabel, labelpad=12)

    ax.set_title(f"Loads on member '{member_name}'")
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)

    fig = ax.figure
    if save_png is not None:
        fig.savefig(save_png, dpi=200, bbox_inches="tight")
    if show_plt:
        plt.show()
    return fig, ax


def plot_model_3d(
    model: FEModel3D,
    *,
    elevation: float = 30,
    azimuth: float = -60,
    roll: float = 0,
    up_axis: UpAxis | str = UpAxis.Y,
    show_node_labels: bool = True,
    show_member_labels: bool = False,
    show_releases: bool = True,
    show_mpcs: bool = True,
    show_plot: bool = True,
    save_png: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    member_color: str = "black",
    mpc_color: str = "magenta",
    node_color: str = "red",
    release_color: str = "dodgerblue",
    node_size: float = 30.0,
    member_lw: float = 1.5,
    mpc_lw: float = 3.0,
    release_marker_size: float = 30.0,
    release_offset_ratio: float = 0.03,
    release_label_offset_ratio: float = 0.03,
    rigid_member_predicate: Callable[[Any], bool] | None = None,
    use_stiffness_rigid_check: bool = False,
    rigid_e_threshold: float = 1.0e9,
    rigid_a_threshold: float = 1.0e6,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot an undeformed PyNite 3D model.

    Args:
        model: PyNite FEModel3D object.
        up_axis: Global axis to display vertically.
        show_node_labels: Display node names.
        show_member_labels: Display member names.
        show_releases: Display member end releases.
        show_mpcs: Display rigid/MPC members.
        show_plot: Display the plot.
        save_png: Optional output PNG path.
        figsize: Matplotlib figure size.
        member_color: Normal member color.
        mpc_color: Rigid/MPC member color.
        node_color: Node marker color.
        release_color: Release marker and label color.
        node_size: Node marker size.
        member_lw: Normal member line width.
        mpc_lw: Rigid/MPC member line width.
        release_marker_size: Release marker size.
        release_offset_ratio: Release marker offset into member as a ratio of model size.
        release_label_offset_ratio: Release label offset as a ratio of model size.
        rigid_member_predicate: Optional custom rigid member detector.
        use_stiffness_rigid_check: Infer rigid members from stiffness values.
        rigid_e_threshold: E threshold for stiffness-based rigid detection.
        rigid_a_threshold: A threshold for stiffness-based rigid detection.

    Returns:
        Matplotlib figure and axes.
    """

    up_axis = _normalize_up_axis(up_axis)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elevation, azim=azimuth, roll=roll)

    normal_segments: list[np.ndarray] = []
    mpc_segments: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    release_endpoint_points: list[tuple[Any, np.ndarray, np.ndarray]] = []

    for member in model.members.values():
        is_rigid = _is_rigid_member(
            member=member,
            rigid_member_predicate=rigid_member_predicate,
            use_stiffness_rigid_check=use_stiffness_rigid_check,
            rigid_e_threshold=rigid_e_threshold,
            rigid_a_threshold=rigid_a_threshold,
        )

        if is_rigid and not show_mpcs:
            continue

        i_plot = _to_plot_xyz(_node_xyz(member.i_node), up_axis)
        j_plot = _to_plot_xyz(_node_xyz(member.j_node), up_axis)

        segment = np.array([i_plot, j_plot], dtype=float)

        if is_rigid:
            mpc_segments.append(segment)
        else:
            normal_segments.append(segment)

        all_points.extend([i_plot, j_plot])
        release_endpoint_points.append((member, i_plot, j_plot))

        if show_member_labels:
            midpoint = 0.5 * (i_plot + j_plot)
            label_color = mpc_color if is_rigid else "blue"
            label_text = f"{member.name} [MPC]" if is_rigid else member.name

            ax.text(
                midpoint[0],
                midpoint[1],
                midpoint[2],
                label_text,
                color=label_color,
                fontsize=8,
            )

    _add_line_collection_3d(
        ax=ax,
        segments=normal_segments,
        color=member_color,
        linewidth=member_lw,
        linestyle="solid",
        alpha=1.0,
    )

    _add_line_collection_3d(
        ax=ax,
        segments=mpc_segments,
        color=mpc_color,
        linewidth=mpc_lw,
        linestyle="dashdot",
        alpha=1.0,
    )

    node_points = np.array(
        [_to_plot_xyz(_node_xyz(node), up_axis) for node in model.nodes.values()],
        dtype=float,
    )

    if len(node_points) > 0:
        ax.scatter(
            node_points[:, 0],
            node_points[:, 1],
            node_points[:, 2],
            color=node_color,
            s=node_size,
            depthshade=True,
        )
        all_points.extend(node_points)

    if show_node_labels:
        for node in model.nodes.values():
            point = _to_plot_xyz(_node_xyz(node), up_axis)

            ax.text(
                point[0],
                point[1],
                point[2],
                node.name,
                fontsize=8,
            )

    model_size = _points_model_size(np.array(all_points, dtype=float))

    if show_releases:
        for member, i_plot, j_plot in release_endpoint_points:
            _plot_member_releases(
                ax=ax,
                member=member,
                i_point=i_plot,
                j_point=j_plot,
                model_size=model_size,
                release_color=release_color,
                release_marker_size=release_marker_size,
                release_offset_ratio=release_offset_ratio,
                release_label_offset_ratio=release_label_offset_ratio,
            )

    _set_axes_equal_3d(ax, np.array(all_points, dtype=float))

    xlabel, ylabel, zlabel = _axis_labels_for_up_axis(up_axis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title("PyNite 3D Model")

    _add_model_legend_handles(
        ax=ax,
        member_color=member_color,
        mpc_color=mpc_color,
        release_color=release_color,
        has_mpcs=bool(mpc_segments),
        show_releases=show_releases,
    )

    fig.tight_layout()

    if save_png is not None:
        fig.savefig(
            save_png,
            dpi=300,
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()

    return fig, ax


def plot_deformed_shape_3d(
    model: FEModel3D,
    combo_name: str,
    *,
    elevation: float = 30,
    azimuth: float = -60,
    roll: float = 0,
    up_axis: UpAxis | str = UpAxis.Y,
    scale_factor: float = 20.0,
    n_points: int = 25,
    show_undeformed: bool = True,
    show_nodes: bool = True,
    show_node_labels: bool = False,
    show_member_labels: bool = False,
    show_releases: bool = True,
    show_mpcs: bool = True,
    show_inactive_tension_only: bool = True,
    show_plot: bool = True,
    save_png: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    member_color: str = "black",
    mpc_color: str = "magenta",
    undeformed_color: str = "0.75",
    undeformed_mpc_color: str = "violet",
    release_color: str = "dodgerblue",
    node_color: str = "black",
    member_lw: float = 2.0,
    mpc_lw: float = 3.0,
    inactive_tension_lw: float = 1.5,
    undeformed_lw: float = 1.0,
    inactive_tension_color: str = "0.25",
    inactive_tension_alpha: float = 0.5,
    inactive_tension_linestyle: str = "solid",
    node_size: float = 20.0,
    release_marker_size: float = 30.0,
    release_offset_ratio: float = 0.03,
    release_label_offset_ratio: float = 0.03,
    curve_released_members: bool = False,
    curve_mpcs: bool = False,
    rigid_member_predicate: Callable[[Any], bool] | None = None,
    use_stiffness_rigid_check: bool = False,
    rigid_e_threshold: float = 1.0e9,
    rigid_a_threshold: float = 1.0e6,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the deformed shape of a PyNite 3D model.

    Args:
        model: PyNite FEModel3D object.
        combo_name: Load combination name to plot.
        up_axis: Global axis to display vertically.
        scale_factor: Displacement scale factor.
        n_points: Number of stations sampled along each member.
        show_undeformed: Plot undeformed geometry.
        show_nodes: Plot deformed node locations.
        show_node_labels: Plot node names at deformed locations.
        show_member_labels: Plot member names near deformed midpoints.
        show_releases: Display member end releases.
        show_mpcs: Display rigid/MPC members.
        show_inactive_tension_only: Display inactive tension-only members.
        show_plot: Display the plot.
        save_png: Optional output PNG path.
        figsize: Matplotlib figure size.
        member_color: Normal deformed member color.
        mpc_color: Rigid/MPC deformed member color.
        inactive_tension_color: Inactive tension-only member color.
        undeformed_color: Normal undeformed member color.
        undeformed_mpc_color: Rigid/MPC undeformed member color.
        release_color: Release marker and label color.
        node_color: Deformed node marker color.
        member_lw: Normal deformed member line width.
        mpc_lw: Rigid/MPC deformed member line width.
        inactive_tension_lw: Inactive tension-only member line width.
        undeformed_lw: Undeformed member line width.
        inactive_tension_alpha: Inactive tension-only transparency.
        inactive_tension_linestyle: Inactive tension-only line style.
        node_size: Deformed node marker size.
        release_marker_size: Release marker size.
        release_offset_ratio: Release marker offset into member as a ratio of model size.
        release_label_offset_ratio: Release label offset as a ratio of model size.
        curve_released_members: Plot released members with member curvature.
        curve_mpcs: Plot MPC members with member curvature.
        rigid_member_predicate: Optional custom rigid member detector.
        use_stiffness_rigid_check: Infer rigid members from stiffness values.
        rigid_e_threshold: E threshold for stiffness-based rigid detection.
        rigid_a_threshold: A threshold for stiffness-based rigid detection.

    Returns:
        Matplotlib figure and axes.
    """

    up_axis = _normalize_up_axis(up_axis)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elevation, azim=azimuth, roll=roll)

    undeformed_segments: list[np.ndarray] = []
    undeformed_mpc_segments: list[np.ndarray] = []
    deformed_segments: list[np.ndarray] = []
    deformed_mpc_segments: list[np.ndarray] = []
    deformed_inactive_tension_segments: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    release_endpoint_points: list[tuple[Any, np.ndarray, np.ndarray]] = []

    for member in model.members.values():
        is_rigid = _is_rigid_member(
            member=member,
            rigid_member_predicate=rigid_member_predicate,
            use_stiffness_rigid_check=use_stiffness_rigid_check,
            rigid_e_threshold=rigid_e_threshold,
            rigid_a_threshold=rigid_a_threshold,
        )

        is_inactive_tension_only = _is_inactive_tension_only_member(
            member=member,
            combo_name=combo_name,
        )

        if is_rigid and not show_mpcs:
            continue

        if is_inactive_tension_only and not show_inactive_tension_only:
            continue

        local_to_global = _member_local_to_global_matrix(
            member=member,
            up_axis=up_axis,
        )

        i_global = _node_xyz(member.i_node)
        length = float(member.L())
        x_stations = np.linspace(0.0, length, n_points)

        undeformed_points_global = np.array(
            [i_global + local_to_global @ np.array([x, 0.0, 0.0]) for x in x_stations],
            dtype=float,
        )

        should_curve_member = not is_inactive_tension_only and _should_plot_member_curvature(
            member=member,
            is_rigid=is_rigid,
            curve_released_members=curve_released_members,
            curve_mpcs=curve_mpcs,
        )

        if should_curve_member:
            deformed_points_global = np.array(
                [
                    _deformed_member_point(
                        member=member,
                        x=x,
                        combo_name=combo_name,
                        scale_factor=scale_factor,
                        local_to_global=local_to_global,
                    )
                    for x in x_stations
                ],
                dtype=float,
            )

            deformed_points_global[0] = _deformed_node_xyz(
                node=member.i_node,
                combo_name=combo_name,
                scale_factor=scale_factor,
            )

            deformed_points_global[-1] = _deformed_node_xyz(
                node=member.j_node,
                combo_name=combo_name,
                scale_factor=scale_factor,
            )

        else:
            i_deformed_global = _deformed_node_xyz(
                node=member.i_node,
                combo_name=combo_name,
                scale_factor=scale_factor,
            )

            j_deformed_global = _deformed_node_xyz(
                node=member.j_node,
                combo_name=combo_name,
                scale_factor=scale_factor,
            )

            deformed_points_global = np.array(
                [
                    i_deformed_global + t * (j_deformed_global - i_deformed_global)
                    for t in np.linspace(0.0, 1.0, n_points)
                ],
                dtype=float,
            )

        undeformed_points = _points_to_plot_xyz(
            points=undeformed_points_global,
            up_axis=up_axis,
        )

        deformed_points = _points_to_plot_xyz(
            points=deformed_points_global,
            up_axis=up_axis,
        )

        if show_undeformed:
            if is_rigid:
                undeformed_mpc_segments.append(undeformed_points)
            else:
                undeformed_segments.append(undeformed_points)

            all_points.extend(undeformed_points)

        if is_inactive_tension_only:
            deformed_inactive_tension_segments.append(deformed_points)
        elif is_rigid:
            deformed_mpc_segments.append(deformed_points)
        else:
            deformed_segments.append(deformed_points)

        all_points.extend(deformed_points)

        release_endpoint_points.append((
            member,
            deformed_points[0],
            deformed_points[-1],
        ))

        if show_member_labels:
            midpoint = deformed_points[len(deformed_points) // 2]

            if is_inactive_tension_only:
                label_color = inactive_tension_color
                label_text = f"{member.name} [Inactive T/O]"
                label_alpha = inactive_tension_alpha
            elif is_rigid:
                label_color = mpc_color
                label_text = f"{member.name} [MPC]"
                label_alpha = 1.0
            else:
                label_color = member_color
                label_text = member.name
                label_alpha = 1.0

            ax.text(
                midpoint[0],
                midpoint[1],
                midpoint[2],
                label_text,
                color=label_color,
                fontsize=8,
                alpha=label_alpha,
            )

    _add_line_collection_3d(
        ax=ax,
        segments=undeformed_segments,
        color=undeformed_color,
        linewidth=undeformed_lw,
        linestyle="dashed",
        alpha=0.8,
    )

    _add_line_collection_3d(
        ax=ax,
        segments=undeformed_mpc_segments,
        color=undeformed_mpc_color,
        linewidth=undeformed_lw,
        linestyle="dashed",
        alpha=0.9,
    )

    _add_line_collection_3d(
        ax=ax,
        segments=deformed_segments,
        color=member_color,
        linewidth=member_lw,
        linestyle="solid",
        alpha=1.0,
    )

    _add_line_collection_3d(
        ax=ax,
        segments=deformed_mpc_segments,
        color=mpc_color,
        linewidth=mpc_lw,
        linestyle="dashdot",
        alpha=1.0,
    )

    _add_line_collection_3d(
        ax=ax,
        segments=deformed_inactive_tension_segments,
        color=inactive_tension_color,
        linewidth=inactive_tension_lw,
        linestyle=inactive_tension_linestyle,
        alpha=inactive_tension_alpha,
    )

    if show_nodes:
        deformed_node_points = []

        for node in model.nodes.values():
            point = _to_plot_xyz(
                point=_deformed_node_xyz(
                    node=node,
                    combo_name=combo_name,
                    scale_factor=scale_factor,
                ),
                up_axis=up_axis,
            )

            deformed_node_points.append(point)
            all_points.append(point)

            if show_node_labels:
                ax.text(
                    point[0],
                    point[1],
                    point[2],
                    node.name,
                    fontsize=8,
                )

        if deformed_node_points:
            deformed_node_points_array = np.array(deformed_node_points, dtype=float)

            ax.scatter(
                deformed_node_points_array[:, 0],
                deformed_node_points_array[:, 1],
                deformed_node_points_array[:, 2],
                color=node_color,
                s=node_size,
                depthshade=True,
            )

    model_size = _points_model_size(np.array(all_points, dtype=float))

    if show_releases:
        for member, i_plot, j_plot in release_endpoint_points:
            _plot_member_releases(
                ax=ax,
                member=member,
                i_point=i_plot,
                j_point=j_plot,
                model_size=model_size,
                release_color=release_color,
                release_marker_size=release_marker_size,
                release_offset_ratio=release_offset_ratio,
                release_label_offset_ratio=release_label_offset_ratio,
            )

    _set_axes_equal_3d(ax, np.array(all_points, dtype=float))

    xlabel, ylabel, zlabel = _axis_labels_for_up_axis(up_axis)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    ax.set_title(f"Deformed Shape\n{combo_name} | Scale = {scale_factor:g}")

    _add_deformed_legend_handles(
        ax=ax,
        member_color=member_color,
        mpc_color=mpc_color,
        undeformed_color=undeformed_color,
        release_color=release_color,
        inactive_tension_color=inactive_tension_color,
        inactive_tension_alpha=inactive_tension_alpha,
        inactive_tension_linestyle=inactive_tension_linestyle,
        show_undeformed=show_undeformed,
        has_mpcs=bool(deformed_mpc_segments),
        has_inactive_tension_only=bool(deformed_inactive_tension_segments),
        show_releases=show_releases,
    )

    fig.tight_layout()

    if save_png is not None:
        fig.savefig(
            save_png,
            dpi=300,
            bbox_inches="tight",
        )

    if show_plot:
        plt.show()

    return fig, ax


def _normalize_up_axis(up_axis: UpAxis | str) -> UpAxis:
    """Normalize an up-axis input to an UpAxis enum.

    Args:
        up_axis: Up axis as an enum or string.

    Returns:
        Normalized UpAxis enum.
    """

    if isinstance(up_axis, UpAxis):
        return up_axis

    if isinstance(up_axis, str):
        try:
            return UpAxis(up_axis.upper())
        except ValueError as exc:
            msg = f"Invalid up_axis {up_axis!r}. Expected X, Y, or Z."
            raise ValueError(msg) from exc

    if hasattr(up_axis, "value"):
        try:
            return UpAxis(str(up_axis.value).upper())
        except ValueError as exc:
            msg = f"Invalid up_axis enum value {up_axis.value!r}."
            raise ValueError(msg) from exc

    msg = f"Invalid up_axis {up_axis!r}. Expected UpAxis or str."
    raise TypeError(msg)


def _to_plot_xyz(
    point: np.ndarray,
    up_axis: UpAxis | str,
) -> np.ndarray:
    """Map global XYZ coordinates to Matplotlib display XYZ coordinates.

    Args:
        point: Global XYZ point.
        up_axis: Global axis to plot vertically.

    Returns:
        Display XYZ point for Matplotlib.
    """

    axis = _normalize_up_axis(up_axis)
    x, y, z = point

    if axis is UpAxis.Z:
        return np.array([x, y, z], dtype=float)

    if axis is UpAxis.Y:
        return np.array([x, z, y], dtype=float)

    return np.array([y, z, x], dtype=float)


def _points_to_plot_xyz(
    points: np.ndarray,
    up_axis: UpAxis | str,
) -> np.ndarray:
    """Map multiple global XYZ points to Matplotlib display XYZ coordinates.

    Args:
        points: Global XYZ points.
        up_axis: Global axis to plot vertically.

    Returns:
        Display XYZ points for Matplotlib.
    """

    return np.array(
        [_to_plot_xyz(point, up_axis) for point in points],
        dtype=float,
    )


def _axis_labels_for_up_axis(
    up_axis: UpAxis | str,
) -> tuple[str, str, str]:
    """Return display axis labels.

    Args:
        up_axis: Global axis to plot vertically.

    Returns:
        Labels for Matplotlib X, Y, and Z axes.
    """

    axis = _normalize_up_axis(up_axis)

    if axis is UpAxis.Z:
        return "X", "Y", "Z"

    if axis is UpAxis.Y:
        return "X", "Z", "Y"

    return "Y", "Z", "X"


def _global_axis_vector(
    axis: UpAxis | str,
) -> np.ndarray:
    """Return a global axis unit vector.

    Args:
        axis: Axis name.

    Returns:
        Axis unit vector.
    """

    normalized_axis = _normalize_up_axis(axis)

    if normalized_axis is UpAxis.X:
        return np.array([1.0, 0.0, 0.0])

    if normalized_axis is UpAxis.Y:
        return np.array([0.0, 1.0, 0.0])

    return np.array([0.0, 0.0, 1.0])


def _candidate_global_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate global axes.

    Returns:
        Global unit axes.
    """

    return (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )


def _node_xyz(node: Node3D) -> np.ndarray:
    """Return a PyNite node's undeformed global coordinates.

    Args:
        node: PyNite node object.

    Returns:
        Node coordinates.
    """

    return np.array(
        [
            float(node.X),
            float(node.Y),
            float(node.Z),
        ],
        dtype=float,
    )


def _deformed_node_xyz(
    node: Node3D,
    combo_name: str,
    scale_factor: float,
) -> np.ndarray:
    """Return a PyNite node's deformed global coordinates.

    Args:
        node: PyNite node object.
        combo_name: Load combination name.
        scale_factor: Displacement scale factor.

    Returns:
        Deformed node coordinates.
    """

    return _node_xyz(node) + scale_factor * np.array(
        [
            _node_displacement(node, "DX", combo_name),
            _node_displacement(node, "DY", combo_name),
            _node_displacement(node, "DZ", combo_name),
        ],
        dtype=float,
    )


def _node_displacement(
    node: Node3D,
    direction: str,
    combo_name: str,
) -> float:
    """Return a nodal displacement.

    Args:
        node: PyNite node object.
        direction: Displacement dictionary name.
        combo_name: Load combination name.

    Returns:
        Nodal displacement value.
    """

    values = getattr(node, direction)

    try:
        return float(values[combo_name])
    except KeyError as exc:
        msg = (
            f"Node {node.name!r} does not have displacement {direction!r} "
            f"for combo {combo_name!r}. Did you analyze the model?"
        )
        raise KeyError(msg) from exc


def _deformed_member_point(
    member: PhysMember,
    x: float,
    combo_name: str,
    scale_factor: float,
    local_to_global: np.ndarray,
) -> np.ndarray:
    """Return the deformed global point on a member at station x.

    Args:
        member: PyNite member object.
        x: Distance from the i-node along the member.
        combo_name: Load combination name.
        scale_factor: Displacement scale factor.
        local_to_global: Local-to-global transformation matrix.

    Returns:
        Deformed global point.
    """

    i_point = _node_xyz(member.i_node)

    local_position = np.array([x, 0.0, 0.0], dtype=float)
    local_deflection = np.array(
        [
            _member_deflection(member, "dx", x, combo_name),
            _member_deflection(member, "dy", x, combo_name),
            _member_deflection(member, "dz", x, combo_name),
        ],
        dtype=float,
    )

    return i_point + local_to_global @ local_position + scale_factor * (local_to_global @ local_deflection)


def _member_deflection(
    member: PhysMember,
    direction: str,
    x: float,
    combo_name: str,
) -> float:
    """Return member deflection at a station.

    Args:
        member: PyNite member object.
        direction: Local deflection direction.
        x: Distance from the i-node along the member.
        combo_name: Load combination name.

    Returns:
        Member deflection.
    """

    try:
        return float(member.deflection(direction, x, combo_name))
    except Exception as exc:
        msg = f"Could not get {direction!r} deflection for member {member.name!r} at x={x:g} for combo {combo_name!r}."
        raise RuntimeError(msg) from exc


def _member_local_to_global_matrix(
    member: PhysMember,
    up_axis: UpAxis | str = UpAxis.Z,
) -> np.ndarray:
    """Return a member local-to-global transformation matrix.

    Args:
        member: PyNite member object.
        up_axis: Global up axis for fallback local axis construction.

    Returns:
        3x3 transformation matrix.
    """

    i_point = _node_xyz(member.i_node)
    j_point = _node_xyz(member.j_node)

    chord = j_point - i_point
    length = np.linalg.norm(chord)

    if length == 0:
        msg = f"Member {member.name!r} has zero length."
        raise ValueError(msg)

    expected_local_x = chord / length

    pynite_transform = _try_get_pynite_transform(member, expected_local_x)

    if pynite_transform is not None:
        return pynite_transform

    return _fallback_local_to_global_matrix(
        member=member,
        up_axis=up_axis,
    )


def _try_get_pynite_transform(
    member: PhysMember,
    expected_local_x: np.ndarray,
) -> np.ndarray | None:
    """Try to get PyNite's local-to-global matrix.

    Args:
        member: PyNite member object.
        expected_local_x: Expected local x-axis in global coordinates.

    Returns:
        3x3 matrix if available, otherwise None.
    """

    for transform_name in ("T", "transformation_matrix"):
        if not hasattr(member, transform_name):
            continue

        transform_item = getattr(member, transform_name)

        try:
            transform = transform_item() if callable(transform_item) else transform_item
            transform = np.array(transform, dtype=float)
        except Exception:
            continue

        if transform.shape[0] < 3 or transform.shape[1] < 3:
            continue

        candidate = transform[:3, :3]

        if _axis_matches(candidate[:, 0], expected_local_x):
            return candidate

        if _axis_matches(candidate[0, :], expected_local_x):
            return candidate.T

    return None


def _fallback_local_to_global_matrix(
    member: PhysMember,
    up_axis: UpAxis | str = UpAxis.Z,
) -> np.ndarray:
    """Build a local-to-global matrix using a configurable up axis.

    Args:
        member: PyNite member object.
        up_axis: Global axis to treat as up.

    Returns:
        3x3 local-to-global transformation matrix.
    """

    i_point = _node_xyz(member.i_node)
    j_point = _node_xyz(member.j_node)

    local_x = _unit_vector(j_point - i_point)
    global_up = _global_axis_vector(up_axis)

    local_z = np.cross(global_up, local_x)

    if np.linalg.norm(local_z) < 1.0e-12:
        for candidate_axis in _candidate_global_axes():
            local_z = np.cross(candidate_axis, local_x)

            if np.linalg.norm(local_z) >= 1.0e-12:
                break

    local_z = _unit_vector(local_z)
    local_y = _unit_vector(np.cross(local_z, local_x))

    rotation_degrees = float(getattr(member, "rotation", 0.0))
    rotation_radians = np.deg2rad(rotation_degrees)

    if abs(rotation_radians) > 0.0:
        local_y = _rotate_vector_about_axis(
            vector=local_y,
            axis=local_x,
            angle=rotation_radians,
        )
        local_z = _rotate_vector_about_axis(
            vector=local_z,
            axis=local_x,
            angle=rotation_radians,
        )

    return np.column_stack((local_x, local_y, local_z))


def _plot_member_releases(
    ax: plt.Axes,
    member: PhysMember,
    i_point: np.ndarray,
    j_point: np.ndarray,
    model_size: float,
    release_color: str,
    release_marker_size: float,
    release_offset_ratio: float,
    release_label_offset_ratio: float,
) -> None:
    """Plot release markers and labels for a member.

    Args:
        ax: Matplotlib 3D axis.
        member: PyNite member object.
        i_point: Plot point for member i-end.
        j_point: Plot point for member j-end.
        model_size: Characteristic model size.
        release_color: Release marker color.
        release_marker_size: Release marker size.
        release_offset_ratio: Release marker offset ratio.
        release_label_offset_ratio: Label offset ratio.
    """

    releases = _member_release_flags(member)

    if not releases.i_releases and not releases.j_releases:
        return

    member_axis = _unit_vector(j_point - i_point)

    release_offset = model_size * release_offset_ratio
    label_offset = model_size * release_label_offset_ratio

    i_release_point = i_point + member_axis * release_offset
    j_release_point = j_point - member_axis * release_offset

    i_label_point = i_release_point - member_axis * label_offset
    j_label_point = j_release_point + member_axis * label_offset

    if releases.i_releases:
        ax.scatter(
            [i_release_point[0]],
            [i_release_point[1]],
            [i_release_point[2]],
            s=release_marker_size,
            marker="o",
            facecolors="none",
            edgecolors=release_color,
            linewidths=1.5,
            depthshade=False,
        )

        ax.text(
            i_label_point[0],
            i_label_point[1],
            i_label_point[2],
            _release_label(releases.i_releases),
            color=release_color,
            fontsize=7,
            ha="center",
            va="center",
        )

    if releases.j_releases:
        ax.scatter(
            [j_release_point[0]],
            [j_release_point[1]],
            [j_release_point[2]],
            s=release_marker_size,
            marker="o",
            facecolors="none",
            edgecolors=release_color,
            linewidths=1.5,
            depthshade=False,
        )

        ax.text(
            j_label_point[0],
            j_label_point[1],
            j_label_point[2],
            _release_label(releases.j_releases),
            color=release_color,
            fontsize=7,
            ha="center",
            va="center",
        )


class _MemberReleaseFlags:
    """Container for member end release labels."""

    def __init__(
        self,
        i_releases: tuple[str, ...],
        j_releases: tuple[str, ...],
    ) -> None:
        """Initialize member release flags.

        Args:
            i_releases: Release labels at member i-end.
            j_releases: Release labels at member j-end.
        """

        self.i_releases = i_releases
        self.j_releases = j_releases


def _member_release_flags(member: PhysMember) -> _MemberReleaseFlags:
    """Return member release flags as readable labels.

    Args:
        member: PyNite member object.

    Returns:
        Release flags for the i-end and j-end.
    """

    release_map = _raw_member_release_map(member)

    i_labels = []
    j_labels = []

    for internal_name, label in (
        ("Dxi", "Dx"),
        ("Dyi", "Dy"),
        ("Dzi", "Dz"),
        ("Rxi", "Rx"),
        ("Ryi", "Ry"),
        ("Rzi", "Rz"),
    ):
        if bool(release_map.get(internal_name, False)):
            i_labels.append(label)

    for internal_name, label in (
        ("Dxj", "Dx"),
        ("Dyj", "Dy"),
        ("Dzj", "Dz"),
        ("Rxj", "Rx"),
        ("Ryj", "Ry"),
        ("Rzj", "Rz"),
    ):
        if bool(release_map.get(internal_name, False)):
            j_labels.append(label)

    return _MemberReleaseFlags(
        i_releases=tuple(i_labels),
        j_releases=tuple(j_labels),
    )


def _raw_member_release_map(member: PhysMember) -> dict[str, bool]:
    """Return raw release flags from common PyNite or wrapper patterns.

    Args:
        member: PyNite member object.

    Returns:
        Dictionary keyed by PyNite-style release names.
    """

    release_names = (
        "Dxi",
        "Dyi",
        "Dzi",
        "Rxi",
        "Ryi",
        "Rzi",
        "Dxj",
        "Dyj",
        "Dzj",
        "Rxj",
        "Ryj",
        "Rzj",
    )

    release_obj = _find_release_object(member)

    if release_obj is None:
        return {name: False for name in release_names}

    if isinstance(release_obj, dict):
        return {
            name: bool(
                release_obj.get(
                    name,
                    release_obj.get(
                        name.lower(),
                        release_obj.get(name.upper(), False),
                    ),
                )
            )
            for name in release_names
        }

    if isinstance(release_obj, (list, tuple, np.ndarray)) and len(release_obj) >= 12:
        return {name: bool(release_obj[index]) for index, name in enumerate(release_names)}

    release_map: dict[str, bool] = {}

    for name in release_names:
        if hasattr(release_obj, name):
            release_map[name] = bool(getattr(release_obj, name))
        elif hasattr(release_obj, name.lower()):
            release_map[name] = bool(getattr(release_obj, name.lower()))
        elif hasattr(member, name):
            release_map[name] = bool(getattr(member, name))
        elif hasattr(member, name.lower()):
            release_map[name] = bool(getattr(member, name.lower()))
        else:
            release_map[name] = False

    return release_map


def _find_release_object(member: PhysMember) -> Any | None:
    """Find a likely release object on a member.

    Args:
        member: PyNite member object.

    Returns:
        Release object if found.
    """

    for attr in (
        "Releases",
        "releases",
        "release",
        "end_releases",
        "EndReleases",
    ):
        if hasattr(member, attr):
            return getattr(member, attr)

    release_names = (
        "Dxi",
        "Dyi",
        "Dzi",
        "Rxi",
        "Ryi",
        "Rzi",
        "Dxj",
        "Dyj",
        "Dzj",
        "Rxj",
        "Ryj",
        "Rzj",
    )

    if any(hasattr(member, name) or hasattr(member, name.lower()) for name in release_names):
        return member

    return None


def _release_label(releases: tuple[str, ...]) -> str:
    """Return a compact release label.

    Args:
        releases: Release labels.

    Returns:
        Compact release label.
    """

    if not releases:
        return ""

    rotational = tuple(release for release in releases if release.startswith("R"))
    translational = tuple(release for release in releases if release.startswith("D"))

    if rotational == ("Rx", "Ry", "Rz") and not translational:
        return "M"

    if set(rotational) == {"Ry", "Rz"} and not translational:
        return "Pin"

    return ",".join(releases)


def _should_plot_member_curvature(
    member: PhysMember,
    is_rigid: bool,
    curve_released_members: bool,
    curve_mpcs: bool,
) -> bool:
    """Return whether a member should be plotted with curved deflection.

    Args:
        member: PyNite member object.
        is_rigid: Whether member is a rigid/MPC member.
        curve_released_members: Plot released members with curved deflection.
        curve_mpcs: Plot MPC members with curved deflection.

    Returns:
        True if member curvature should be plotted.
    """

    if is_rigid:
        return curve_mpcs

    if curve_released_members:
        return True

    releases = _member_release_flags(member)

    if _is_pin_pin_member(releases):
        return False

    if _has_many_rotational_releases(releases):
        return False

    return True


def _is_pin_pin_member(
    releases: _MemberReleaseFlags,
) -> bool:
    """Return whether a member is released like a pin-pin brace.

    Args:
        releases: Member release flags.

    Returns:
        True if both ends are released for local bending.
    """

    i_releases = set(releases.i_releases)
    j_releases = set(releases.j_releases)

    i_is_pin = {"Ry", "Rz"}.issubset(i_releases)
    j_is_pin = {"Ry", "Rz"}.issubset(j_releases)

    return i_is_pin and j_is_pin


def _has_many_rotational_releases(
    releases: _MemberReleaseFlags,
) -> bool:
    """Return whether a member has enough releases to avoid curvature plotting.

    Args:
        releases: Member release flags.

    Returns:
        True if member curvature plotting should be avoided.
    """

    rotational_releases = {"Rx", "Ry", "Rz"}

    i_count = len(set(releases.i_releases) & rotational_releases)
    j_count = len(set(releases.j_releases) & rotational_releases)

    return i_count + j_count >= 4


def _is_inactive_tension_only_member(
    member: PhysMember,
    combo_name: str,
) -> bool:
    """Return whether a tension-only member is inactive for a load combo.

    Args:
        member: PyNite member object.
        combo_name: Load combination name.

    Returns:
        True if the member appears to be tension-only and inactive.
    """

    if not _is_tension_only_member(member):
        return False

    active_value = _get_combo_value_from_possible_attrs(
        obj=member,
        attr_names=(
            "active",
            "Active",
            "is_active",
            "IsActive",
        ),
        combo_name=combo_name,
    )

    if active_value is not None:
        return not bool(active_value)

    inactive_value = _get_combo_value_from_possible_attrs(
        obj=member,
        attr_names=(
            "inactive",
            "Inactive",
            "is_inactive",
            "IsInactive",
        ),
        combo_name=combo_name,
    )

    if inactive_value is not None:
        return bool(inactive_value)

    axial_value = _try_member_axial_at_midspan(
        member=member,
        combo_name=combo_name,
    )

    if axial_value is None:
        return False

    return axial_value < 0.0


def _is_tension_only_member(member: PhysMember) -> bool:
    """Return whether a member is tension-only.

    Args:
        member: PyNite member object.

    Returns:
        True if member appears to be tension-only.
    """

    for attr in (
        "tension_only",
        "tensionOnly",
        "TensionOnly",
        "tension_only_member",
        "is_tension_only",
    ):
        if not hasattr(member, attr):
            continue

        value = getattr(member, attr)

        if isinstance(value, bool):
            return value

        if callable(value):
            try:
                return bool(value())
            except TypeError:
                pass

    for attr in (
        "member_type",
        "type",
        "kind",
        "classification",
    ):
        if hasattr(member, attr):
            value_name = _normalized_type_name(getattr(member, attr))

            if value_name in {
                "TENSION_ONLY",
                "TENSIONONLY",
                "TENSION_ONLY_MEMBER",
            }:
                return True

    return False


def _get_combo_value_from_possible_attrs(
    obj: Any,
    attr_names: tuple[str, ...],
    combo_name: str,
) -> Any | None:
    """Return a combo-specific value from possible object attributes.

    Args:
        obj: Object to inspect.
        attr_names: Attribute names to try.
        combo_name: Load combination name.

    Returns:
        Combo-specific value if found, otherwise None.
    """

    for attr_name in attr_names:
        if not hasattr(obj, attr_name):
            continue

        value = getattr(obj, attr_name)

        if callable(value):
            try:
                value = value()
            except TypeError:
                try:
                    value = value(combo_name)
                except TypeError:
                    continue

        if isinstance(value, dict):
            if combo_name in value:
                return value[combo_name]

            for key, keyed_value in value.items():
                if str(key) == combo_name:
                    return keyed_value

            continue

        if isinstance(value, bool):
            return value

    return None


def _try_member_axial_at_midspan(
    member: PhysMember,
    combo_name: str,
) -> float | None:
    """Try to get member axial force at midspan.

    Args:
        member: PyNite member object.
        combo_name: Load combination name.

    Returns:
        Axial value if available, otherwise None.
    """

    try:
        length = float(member.L())
        return float(
            member.axial(
                x=0.5 * length,
                combo_name=combo_name,
            )
        )
    except Exception:
        pass

    try:
        length = float(member.L())
        return float(member.axial(0.5 * length, combo_name))
    except Exception:
        return None


def _is_rigid_member(
    member: PhysMember,
    rigid_member_predicate: Callable[[Any], bool] | None,
    use_stiffness_rigid_check: bool,
    rigid_e_threshold: float,
    rigid_a_threshold: float,
) -> bool:
    """Return whether a member should be treated as rigid/MPC.

    Args:
        member: PyNite member object.
        rigid_member_predicate: Optional user-provided rigid-member checker.
        use_stiffness_rigid_check: Check stiffness values to infer rigidity.
        rigid_e_threshold: E threshold for stiffness-based rigid detection.
        rigid_a_threshold: A threshold for stiffness-based rigid detection.

    Returns:
        True if the member should be plotted as a rigid/MPC member.
    """

    if rigid_member_predicate is not None:
        return bool(rigid_member_predicate(member))

    for attr in ("is_rigid", "is_mpc", "is_mpc_member", "rigid", "mpc"):
        if hasattr(member, attr):
            value = getattr(member, attr)

            if isinstance(value, bool):
                return value

            if callable(value):
                try:
                    return bool(value())
                except TypeError:
                    pass

    for attr in ("member_type", "type", "kind", "classification"):
        if hasattr(member, attr):
            value_name = _normalized_type_name(getattr(member, attr))

            if value_name in {"MPC", "RIGID", "RIGID_LINK", "RIGID_MEMBER"}:
                return True

    for attr in ("name", "material_name", "section_name"):
        if hasattr(member, attr):
            value_name = str(getattr(member, attr)).upper()

            if any(token in value_name for token in ("MPC", "RIGID", "LINK")):
                return True

    if use_stiffness_rigid_check:
        return _is_rigid_by_stiffness(
            member=member,
            rigid_e_threshold=rigid_e_threshold,
            rigid_a_threshold=rigid_a_threshold,
        )

    return False


def _normalized_type_name(value: Any) -> str:
    """Return a normalized type string.

    Args:
        value: Type-like value.

    Returns:
        Normalized uppercase type string.
    """

    if hasattr(value, "name"):
        return str(value.name).upper()

    return str(value).upper().replace(" ", "_").replace("-", "_")


def _is_rigid_by_stiffness(
    member: PhysMember,
    rigid_e_threshold: float,
    rigid_a_threshold: float,
) -> bool:
    """Infer rigid behavior from member stiffness values.

    Args:
        member: PyNite member object.
        rigid_e_threshold: E threshold for rigid detection.
        rigid_a_threshold: A threshold for rigid detection.

    Returns:
        True if member stiffness appears intentionally rigid.
    """

    e_value = _get_nested_numeric_attr(
        member,
        attr_paths=(
            ("material", "E"),
            ("Material", "E"),
            ("E",),
        ),
    )

    a_value = _get_nested_numeric_attr(
        member,
        attr_paths=(
            ("section", "A"),
            ("Section", "A"),
            ("A",),
        ),
    )

    if e_value is None or a_value is None:
        return False

    return e_value >= rigid_e_threshold and a_value >= rigid_a_threshold


def _get_nested_numeric_attr(
    obj: Any,
    attr_paths: tuple[tuple[str, ...], ...],
) -> float | None:
    """Get a nested numeric attribute.

    Args:
        obj: Object to inspect.
        attr_paths: Attribute paths to try.

    Returns:
        Numeric value if resolved successfully, otherwise None.
    """

    for attr_path in attr_paths:
        current = obj

        try:
            for attr in attr_path:
                current = getattr(current, attr)

            return float(current)
        except Exception:
            continue

    return None


def _add_line_collection_3d(
    ax: plt.Axes,
    segments: list[np.ndarray],
    color: str,
    linewidth: float,
    linestyle: str,
    alpha: float,
) -> None:
    """Add 3D line segments to an axis.

    Args:
        ax: Matplotlib 3D axis.
        segments: Line segments.
        color: Line color.
        linewidth: Line width.
        linestyle: Line style.
        alpha: Line transparency.
    """

    if not segments:
        return

    collection = Line3DCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
    )

    ax.add_collection3d(collection)


def _axis_matches(
    candidate_axis: np.ndarray,
    expected_axis: np.ndarray,
    tolerance: float = 1.0e-6,
) -> bool:
    """Check whether two axes point in the same direction.

    Args:
        candidate_axis: Candidate axis.
        expected_axis: Expected axis.
        tolerance: Numerical tolerance.

    Returns:
        True if axes match.
    """

    candidate_axis = _unit_vector(candidate_axis)
    expected_axis = _unit_vector(expected_axis)

    return bool(np.linalg.norm(candidate_axis - expected_axis) <= tolerance)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector.

    Args:
        vector: Input vector.

    Returns:
        Unit vector.
    """

    norm = np.linalg.norm(vector)

    if norm == 0:
        msg = "Cannot normalize a zero-length vector."
        raise ValueError(msg)

    return vector / norm


def _rotate_vector_about_axis(
    vector: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate a vector about an axis.

    Args:
        vector: Vector to rotate.
        axis: Rotation axis.
        angle: Rotation angle in radians.

    Returns:
        Rotated vector.
    """

    axis = _unit_vector(axis)

    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def _set_axes_equal_3d(
    ax: plt.Axes,
    points: np.ndarray,
) -> None:
    """Set equal scale for a 3D Matplotlib axis.

    Args:
        ax: Matplotlib 3D axis.
        points: Points to include in axis bounds.
    """

    if points.size == 0:
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    ranges = maxs - mins
    max_range = float(np.max(ranges))

    if max_range == 0:
        max_range = 1.0

    centers = (mins + maxs) / 2.0
    half_range = max_range / 2.0

    ax.set_xlim(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim(centers[2] - half_range, centers[2] + half_range)


def _points_model_size(points: np.ndarray) -> float:
    """Return characteristic model size from points.

    Args:
        points: XYZ points.

    Returns:
        Characteristic model size.
    """

    if points.size == 0:
        return 1.0

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    model_size = float(np.max(ranges))

    if model_size == 0:
        return 1.0

    return model_size


def _add_model_legend_handles(
    ax: plt.Axes,
    member_color: str,
    mpc_color: str,
    release_color: str,
    has_mpcs: bool,
    show_releases: bool,
) -> None:
    """Add legend handles for model plot.

    Args:
        ax: Matplotlib axis.
        member_color: Normal member color.
        mpc_color: MPC member color.
        release_color: Release marker color.
        has_mpcs: Whether MPC members were plotted.
        show_releases: Whether releases are shown.
    """

    ax.plot(
        [],
        [],
        [],
        color=member_color,
        lw=1.5,
        label="Members",
    )

    if has_mpcs:
        ax.plot(
            [],
            [],
            [],
            color=mpc_color,
            lw=3.0,
            linestyle="dashdot",
            label="Rigid / MPC members",
        )

    if show_releases:
        ax.scatter(
            [],
            [],
            [],
            s=65.0,
            marker="o",
            facecolors="white",
            edgecolors=release_color,
            linewidths=1.5,
            label="Member release",
        )

    ax.legend()


def _add_deformed_legend_handles(
    ax: plt.Axes,
    member_color: str,
    mpc_color: str,
    undeformed_color: str,
    release_color: str,
    inactive_tension_color: str,
    inactive_tension_alpha: float,
    inactive_tension_linestyle: str,
    show_undeformed: bool,
    has_mpcs: bool,
    has_inactive_tension_only: bool,
    show_releases: bool,
) -> None:
    """Add simple legend handles for the deformed shape plot.

    Args:
        ax: Matplotlib axis.
        member_color: Normal member color.
        mpc_color: MPC member color.
        undeformed_color: Undeformed member color.
        release_color: Release marker color.
        inactive_tension_color: Inactive tension-only member color.
        inactive_tension_alpha: Inactive tension-only alpha.
        inactive_tension_linestyle: Inactive tension-only line style.
        show_undeformed: Whether undeformed geometry is shown.
        has_mpcs: Whether MPC members were plotted.
        has_inactive_tension_only: Whether inactive tension-only members were plotted.
        show_releases: Whether releases are shown.
    """

    ax.plot(
        [],
        [],
        [],
        color=member_color,
        lw=2.0,
        label="Deformed members",
    )

    if has_mpcs:
        ax.plot(
            [],
            [],
            [],
            color=mpc_color,
            lw=3.0,
            linestyle="dashdot",
            label="Rigid / MPC members",
        )

    if has_inactive_tension_only:
        ax.plot(
            [],
            [],
            [],
            color=inactive_tension_color,
            lw=1.5,
            linestyle=inactive_tension_linestyle,
            alpha=inactive_tension_alpha,
            label="Inactive tension-only members",
        )

    if show_undeformed:
        ax.plot(
            [],
            [],
            [],
            color=undeformed_color,
            lw=1.0,
            linestyle="dashed",
            label="Undeformed geometry",
        )

    if show_releases:
        ax.scatter(
            [],
            [],
            [],
            s=65.0,
            marker="o",
            facecolors="white",
            edgecolors=release_color,
            linewidths=1.5,
            label="Member release",
        )

    ax.legend()
