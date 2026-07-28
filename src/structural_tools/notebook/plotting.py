"""Plot functions"""

from __future__ import annotations

from collections import defaultdict
from math import isclose

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from Pynite import PhysMember

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
