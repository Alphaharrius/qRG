"""Matplotlib, plotly, and networkx renderers for a node tree."""

from __future__ import annotations

import textwrap
from typing import Any, cast

import matplotlib.patheffects as pe
import networkx as nx  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from .tree import OPERATOR, Node


def _tree_layout(root: Node) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    next_x = 0.0

    def assign(node: Node, depth: int) -> float:
        nonlocal next_x
        children = list(node.children.values())
        if not children:
            x = next_x
            next_x += 1.0
        else:
            child_xs = [assign(child, depth + 1) for child in children]
            x = sum(child_xs) / len(child_xs)
        positions[id(node)] = (x, -float(depth))
        return x

    assign(root, 0)
    return positions


def _node_tag(node: Node) -> str:
    step = node.step
    if node.failed:
        return f"ERROR: {step.method}" if step is not None else "ERROR"
    if step is None:
        return OPERATOR
    return step.method


def _wrap_node_label(label: str, tag: str, *, width: int) -> str:
    text = f"{label}\n[{tag}]"
    return "\n".join(textwrap.fill(line, width=width) for line in text.splitlines())


def _tree_node_style(
    *, is_current: bool, is_root: bool, is_failed: bool = False
) -> dict[str, object]:
    if is_failed:
        return {
            "facecolor": "#fee2e2" if not is_current else "#b91c1c",
            "edgecolor": "#991b1b",
            "textcolor": "#7f1d1d" if not is_current else "white",
            "linewidth": 2.4,
        }
    if is_current:
        return {
            "facecolor": "#d1495b",
            "edgecolor": "#7f1d1d",
            "textcolor": "white",
            "linewidth": 2.2,
        }
    if is_root:
        return {
            "facecolor": "#edc948",
            "edgecolor": "#8c6d1f",
            "textcolor": "black",
            "linewidth": 2.0,
        }
    return {
        "facecolor": "#d6ebf5",
        "edgecolor": "#355070",
        "textcolor": "#1f2933",
        "linewidth": 1.6,
    }


@Node.register_plot_method("tree", backend="networkx")  # type: ignore[untyped-decorator]
def _plot_node_tree_networkx(node: Node) -> nx.DiGraph:
    """
    Return a directed graph of the whole tree, rooted at ``node.root()``.

    The current node is flagged in node attributes. Layout positions live in
    ``graph["positions"]``.
    """
    root = node.root()
    positions = _tree_layout(root)
    graph = nx.DiGraph()
    graph.graph["root"] = root
    graph.graph["current"] = node
    graph.graph["positions"] = positions

    stack = [root]
    while stack:
        current = stack.pop()
        current_id = id(current)
        graph.add_node(
            current_id,
            node=current,
            label=current.name,
            tag=_node_tag(current),
            is_current=current is node,
            is_root=current is root,
            is_failed=current.failed,
        )
        for key, child in current.children.items():
            step = child.step
            if step is None:
                raise RuntimeError(f"Child {child.name!r} has no step")
            child_id = id(child)
            graph.add_node(
                child_id,
                node=child,
                label=child.name,
                tag=_node_tag(child),
                is_current=child is node,
                is_root=False,
                is_failed=child.failed,
            )
            graph.add_edge(current_id, child_id, target=key, transform=step.transform)
            stack.append(child)
    return graph


@Node.register_plot_method("tree", backend="matplotlib")  # type: ignore[untyped-decorator]
def _plot_node_tree_matplotlib(
    node: Node,
    *,
    ax: Any = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    label_width: int = 18,
) -> Any:
    """Draw the tree with wrapped labels. The current node is crimson."""
    graph = _plot_node_tree_networkx(node)
    positions = graph.graph["positions"]

    created_fig = False
    if ax is None:
        fig = Figure(figsize=figsize)
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        created_fig = True
    else:
        fig = ax.figure

    fig.patch.set_facecolor("#f7f3ea")
    ax.set_facecolor("#f7f3ea")

    for src, dst in graph.edges:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        y_start = y0 - 0.16
        junction_y = y0 - 0.42
        y_end = y1 + 0.04
        connector = PathPatch(
            MplPath(
                [(x0, y_start), (x0, junction_y), (x1, junction_y), (x1, y_end)],
                [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
            ),
            facecolor="none",
            edgecolor="#6b7a8f",
            linewidth=1.8,
            capstyle="round",
            joinstyle="round",
            zorder=2,
            alpha=0.9,
        )
        connector.set_path_effects([pe.Stroke(linewidth=3.2, foreground="#ebe4d8"), pe.Normal()])
        ax.add_patch(connector)

    for node_id, attrs in graph.nodes(data=True):
        x, y = positions[node_id]
        style = _tree_node_style(
            is_current=attrs["is_current"],
            is_root=attrs["is_root"],
            is_failed=attrs.get("is_failed", False),
        )
        label = _wrap_node_label(attrs["label"], attrs["tag"], width=label_width)
        text = ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color=style["textcolor"],
            fontsize=10,
            fontweight="bold",
            fontfamily="monospace",
            bbox={
                "boxstyle": "round,pad=0.55,rounding_size=0.25",
                "facecolor": style["facecolor"],
                "edgecolor": style["edgecolor"],
                "linewidth": style["linewidth"],
            },
            zorder=3,
        )
        text.set_path_effects([pe.withStroke(linewidth=0.5, foreground=style["facecolor"])])

    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    ax.set_xlim(min(xs) - 0.95, max(xs) + 0.95)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.set_axis_off()
    ax.set_title(
        f"Tree View: {graph.graph['current'].name}",
        fontsize=13,
        fontweight="bold",
        color="#243447",
        pad=18,
    )
    if created_fig:
        fig.tight_layout()
    return fig


@Node.register_plot_method("tree", backend="plotly")  # type: ignore[untyped-decorator]
def _plot_node_tree_plotly(
    node: Node,
    *,
    figure: Any = None,
    label_width: int = 18,
) -> Any:
    """Draw the tree as an interactive plotly figure."""
    graph = _plot_node_tree_networkx(node)
    positions = graph.graph["positions"]
    fig = go.Figure() if figure is None else figure

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for src, dst in graph.edges:
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        y_start = y0 - 0.16
        junction_y = y0 - 0.42
        y_end = y1 + 0.04
        edge_x.extend([x0, x0, x1, x1, None])
        edge_y.extend([y_start, junction_y, junction_y, y_end, None])

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"color": "#6b7a8f", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    node_x: list[float] = []
    node_y: list[float] = []
    hover_text: list[str] = []
    marker_color: list[str] = []
    marker_line_color: list[str] = []
    marker_line_width: list[float] = []
    for node_id, attrs in graph.nodes(data=True):
        x, y = positions[node_id]
        style = _tree_node_style(
            is_current=attrs["is_current"],
            is_root=attrs["is_root"],
            is_failed=attrs.get("is_failed", False),
        )
        label = _wrap_node_label(
            cast(str, attrs["label"]),
            cast(str, attrs["tag"]),
            width=label_width,
        )
        fig.add_annotation(
            x=x,
            y=y,
            text=label.replace("\n", "<br>"),
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            align="center",
            font={"color": style["textcolor"], "size": 11, "family": "Courier New, monospace"},
            bgcolor=style["facecolor"],
            bordercolor=style["edgecolor"],
            borderwidth=style["linewidth"],
            borderpad=10,
        )
        node_x.append(x)
        node_y.append(y)
        hover_text.append(f"name: {attrs['label']}<br>step: {attrs['tag']}")
        marker_color.append(cast(str, style["facecolor"]))
        marker_line_color.append(cast(str, style["edgecolor"]))
        marker_line_width.append(cast(float, style["linewidth"]))

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker={
                "size": 18,
                "color": marker_color,
                "line": {"color": marker_line_color, "width": marker_line_width},
                "opacity": 0.0,
            },
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    fig.update_layout(
        title={
            "text": f"Tree View: {graph.graph['current'].name}",
            "font": {"size": 18, "color": "#243447"},
            "x": 0.5,
        },
        paper_bgcolor="#f7f3ea",
        plot_bgcolor="#f7f3ea",
        xaxis={"range": [min(xs) - 0.95, max(xs) + 0.95], "visible": False},
        yaxis={
            "range": [min(ys) - 0.9, max(ys) + 0.9],
            "visible": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig
