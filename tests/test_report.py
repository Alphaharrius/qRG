from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from qrg.strat import register, unregister
from qrg.tree import Branch, Node, Strategy


class FakeTensor:
    def __init__(self, name: str, shape: tuple[int, ...] = (8, 4, 4)) -> None:
        self.name = name
        self.data = type("FakeData", (), {"shape": shape, "ndim": len(shape)})()
        self.dims = ()

    def h(self, *_axes: int) -> FakeTensor:
        return self

    def __matmul__(self, _other: FakeTensor) -> FakeTensor:
        return FakeTensor(f"{self.name}_proj", self.data.shape)


def _fake(name: str, shape: tuple[int, ...] = (8, 4, 4)) -> Any:
    return FakeTensor(name, shape)


def _extract_payload(html_text: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="tree-data" type="application/json">(.*?)</script>',
        html_text,
        re.S,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_report_embeds_the_step_on_the_link_and_the_record_on_the_node(
    tmp_path: Path,
) -> None:
    def impl(node: Node, **params: Any) -> Branch:
        fig_f = go.Figure(go.Scatter(x=[0, 1], y=[-1.0, -0.5]))
        fig_c_3d = go.Figure(go.Scatter3d(x=[0.0], y=[0.0], z=[0.0], mode="markers"))
        return Branch(
            data={
                "H": _fake("courier_H", (8, 4, 4)),
                "frozen": _fake("Wf", (8, 108, 104)),
                "courier": _fake("Wc", (8, 108, 4)),
            },
            info={
                "factor": params["factor"],
                "frozen": {
                    "overlap": 0.91,
                    "z2": (0, 0, 0, 0),
                    "direct_gap": 1.15,
                    "td_site_irreps": {
                        "filled": {"Γ6": "3 copies", "Γ7": "2 copies", "Γ8": "4 copies"}
                    },
                    "plots": {"Frozen Bands": fig_f},
                },
                "courier": {
                    "overlap": 0.37,
                    "z2": (1, 1, 1, 1),
                    "direct_gap": 1.46,
                    "plots": {"3D Courier N(k)": fig_c_3d},
                },
            },
            kind="isometry",
        )

    register("report-blocking", impl)
    root = Node.new(name="lattice", operator=_fake("H", (216, 4, 4)))
    root.record("z2", (1, 0, 0, 0))
    root.record(
        "rg_flow_summary",
        [{"stage": "Cycle 0", "k_points": 216, "direct_gap": 0.99}],
    )
    try:
        root.add_strat("blocking", Strategy("report-blocking", {"factor": 2}))
        root.grow("blocking")
        page = root.report(tmp_path / "tree.html")
    finally:
        unregister("report-blocking")

    text = page.read_text(encoding="utf-8")
    assert "report-blocking  factor 2" in text
    assert "frozen isometry" in text
    assert "courier isometry" in text
    assert '"selected": "blocking"' in text
    assert "[1, 0, 0, 0]" in text

    payload = _extract_payload(text)
    by_id = {item["id"]: item for item in payload["nodes"]}
    assert "blocking/frozen" in by_id
    assert by_id["blocking"]["frozen_id"] == "blocking/frozen"
    assert by_id["blocking/frozen"]["role"] == "frozen_sibling"
    assert by_id["blocking"]["role"] == "courier"
    assert by_id["blocking/frozen"]["plots"][0]["title"] == "Frozen Bands"
    assert by_id["blocking"]["plots"][0]["title"] == "3D Courier N(k)"
    assert by_id["lattice"]["info"]["rg_flow_summary"][0]["stage"] == "Cycle 0"


def test_report_renders_error_block_and_auto_selects_it(tmp_path: Path) -> None:
    def boom(node: Node, **_params: Any) -> Branch:
        raise RuntimeError("Multiplet split at -26 during Wannier step")

    register("report-boom", boom)
    root = Node.new(name="diamond", operator=_fake("H", (216, 4, 4)))
    try:
        root.add_strat("blocking", Strategy("report-boom", {"factor": 3}))
        err_node = root.grow("blocking", on_error="record")
        assert err_node.failed is True
        page = root.report(tmp_path / "error_tree.html")
    finally:
        unregister("report-boom")

    text = page.read_text(encoding="utf-8")
    payload = _extract_payload(text)
    assert payload["selected"] == "blocking"
    by_id = {item["id"]: item for item in payload["nodes"]}
    err_entry = by_id["blocking"]
    assert err_entry["role"] == "error"
    assert err_entry["label"] == "ERROR (blocking)"
    assert err_entry["badge"] == "ERROR · RuntimeError"
    assert "Multiplet split at -26" in err_entry["error"]["message"]
    assert "RuntimeError" in err_entry["error"]["traceback"]
