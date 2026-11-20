from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import pytest

import qrg.tree as tree_module
from qrg.strat import register, unregister
from qrg.tree import Branch, Node, Strategy


class FakeTensor:
    def __init__(
        self, name: str, dims: tuple[str, str] = ("d0", "d1"), device: str = "cpu"
    ) -> None:
        self.name = name
        self.dims = dims
        self.device = device

    def __matmul__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(f"({self.name}@{other.name})", dims=self.dims, device=self.device)

    def h(self, *_axes: int) -> FakeTensor:
        return FakeTensor(f"{self.name}.h", dims=self.dims, device=self.device)


def _tensor(name: str, **kwargs: Any) -> Any:
    return FakeTensor(name, **kwargs)


def _install_impl() -> None:
    def impl(node: Node, **params: Any) -> Branch:
        label = str(params.get("label", "H"))
        kind = str(params.get("kind", "isometry"))
        transform = None if kind != "isometry" else _tensor(str(params.get("edge", "W")))
        return Branch(
            data={"H": _tensor(label, dims=params.get("dims", ("d0", "d1")), device="cpu")},
            transform=transform,
            info=dict(params),
            kind=kind,
            name=params.get("name"),
        )

    register("unit-pass", impl)


def _strat(params: dict[str, Any]) -> Strategy:
    return Strategy("unit-pass", params)


@pytest.fixture(autouse=True)
def _unit_impl() -> Any:
    _install_impl()
    yield
    unregister("unit-pass")


def test_grow_returns_the_new_layer_and_keeps_its_tensors() -> None:
    root = Node.new(name="root", operator=_tensor("root"))
    root.add_strat("blocking", _strat({"label": "folded", "kind": "blocking", "factor": 3}))
    root.add_strat("wannier", _strat({"label": "courier", "edge": "Wc", "name": "Wannier"}))

    layer = root.grow("blocking").grow("wannier")

    assert layer.path() == "blocking.wannier"
    assert layer.name == "Wannier"
    assert cast(FakeTensor, layer.operator()).name == "courier"
    assert cast(FakeTensor, root.trace("blocking").operator()).name == "folded"
    assert layer.step is not None
    assert layer.step.method == "wannier"
    assert layer.step.impl == "unit-pass"
    assert layer.step.params["edge"] == "Wc"
    assert layer.parent().parent() is root
    assert "blocking" in layer.strats


def test_add_strat_from_a_child_updates_the_root() -> None:
    root = Node.new(name="root", operator=_tensor("root"))
    root.add_strat("blocking", _strat({"label": "folded", "kind": "blocking"}))
    blocked = root.grow("blocking")
    blocked.add_strat("wannier", _strat({"label": "courier"}))

    assert blocked.grow("wannier").path() == "blocking.wannier"
    assert root.strats["wannier"].impl == "unit-pass"


def test_grow_rejects_missing_and_duplicate_strategies() -> None:
    root = Node.new(name="root", operator=_tensor("root"))
    with pytest.raises(ValueError, match="not set"):
        root.grow("blocking")
    with pytest.raises(ValueError, match="cannot be empty or contain"):
        root.add_strat("a.b", _strat({}))
    root.add_strat("blocking", _strat({"kind": "blocking"}))
    root.grow("blocking")
    with pytest.raises(ValueError, match="already has a branch"):
        root.grow("blocking")


def test_chain_can_repeat_the_same_strategy_names() -> None:
    root = Node.new(name="root", operator=_tensor("root"))
    root.add_strat("blocking", _strat({"label": "folded", "kind": "blocking"}))
    root.add_strat("wannier", _strat({"label": "courier"}))

    layer = root
    for _ in range(2):
        layer = layer.grow("blocking").grow("wannier")

    assert layer.path() == "blocking.wannier.blocking.wannier"
    assert root.trace("blocking.wannier.blocking").info["kind"] == "blocking"


def test_compute_and_record_store_on_the_node() -> None:
    derived = _tensor("derived")
    root = Node.new(name="root", operator=_tensor("root"))
    root.compute("copy", lambda node: derived).record("z2", (1, 0, 0, 0))

    assert root["copy"] is derived
    assert root.info["z2"] == (1, 0, 0, 0)


def test_cut_copies_strategies_onto_the_detached_root() -> None:
    root = Node.new(name="root", operator=_tensor("root"))
    root.add_strat("blocking", _strat({"label": "folded", "kind": "blocking"}))
    blocked = root.grow("blocking")
    detached = root.cut("blocking")

    assert detached.is_root() is True
    assert detached.step is None
    assert "blocking" in detached.strats
    assert detached.grow("blocking").path() == "blocking"
    with pytest.raises(ValueError, match="Target blocking not found"):
        root.leaf("blocking")
    assert blocked is detached


def test_trace_path_and_find() -> None:
    root = Node.new(name="root", operator=_tensor("root"))
    root.add_strat("alpha", _strat({"label": "A", "name": "alpha-child"}))
    root.add_strat("beta", _strat({"label": "B", "name": "beta-child"}))
    root.add_strat("gamma", _strat({"label": "G", "name": "gamma-child"}))
    root.grow("alpha")
    root.grow("beta")
    alpha = root.leaf("alpha").node
    beta = root.leaf("beta").node
    alpha.grow("gamma")
    gamma = alpha.leaf("gamma").node

    assert root.trace("") is root
    assert alpha.trace(".") is alpha
    assert root.trace("alpha.gamma") is gamma
    assert gamma.path() == "alpha.gamma"
    assert root.find(regex="child$") == [alpha, gamma, beta]
    with pytest.raises(ValueError, match=r'trace cannot be resolve at "alpha\.\[missing\]"'):
        root.trace("alpha.missing")
    with pytest.raises(ValueError, match="requires regex and/or predicate"):
        root.find()


def test_get_transform_composes_isometries_and_stops_at_a_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Node.new(
        name="root",
        operator=_tensor("root", dims=("parent", "parent"), device="cpu"),
    )
    root.add_strat(
        "blocking", _strat({"label": "folded", "kind": "blocking", "dims": ("block", "block")})
    )
    root.add_strat(
        "wannier", _strat({"label": "courier", "edge": "Wc", "dims": ("courier", "courier")})
    )
    blocked = root.grow("blocking")
    blocked.grow("wannier")

    seen: dict[str, Any] = {}

    def eye(dims: Any, *, device: Any = None) -> FakeTensor:
        seen["dims"] = dims
        seen["device"] = device
        return FakeTensor("identity", dims=dims, device=device or "cpu")

    monkeypatch.setattr(tree_module.qten, "eye", eye)
    assert cast(FakeTensor, blocked.get_transform(".wannier")).name == "Wc"
    assert cast(FakeTensor, blocked.get_transform(".")).name == "identity"
    assert seen["dims"] == ("block", "block")
    with pytest.raises(ValueError, match="step kind is 'blocking'"):
        root.get_transform("blocking.wannier")


def test_pickle_keeps_strategies_and_layers() -> None:
    root = Node.new(name="root", operator=_tensor("root"))
    root.add_strat("blocking", _strat({"label": "folded", "kind": "blocking", "factor": 3}))
    root.grow("blocking")
    restored = pickle.loads(pickle.dumps(root))

    assert restored.strats["blocking"].params["factor"] == 3
    blocked = restored.trace("blocking")
    assert blocked.parent() is restored
    assert cast(FakeTensor, blocked.operator()).name == "folded"
    restored.add_strat("wannier", _strat({"label": "courier"}))
    assert blocked.grow("wannier").path() == "blocking.wannier"


def test_trace_accepts_absolute_and_relative_layer_counts() -> None:
    root = Node.new(name="root", operator=_tensor("root"))
    root.add_strat("blocking", _strat({"label": "folded", "kind": "blocking"}))
    root.add_strat("wannier", _strat({"label": "courier"}))
    blocked = root.grow("blocking")
    layer = blocked.grow("wannier")

    assert layer.depth() == 2
    assert root.trace(0) is root
    assert root.trace(2) is layer
    assert layer.trace(0, relative=True) is layer
    assert layer.trace(-1) is blocked
    assert layer.trace(-2) is root
    assert blocked.trace(1, relative=True) is layer
    assert blocked.trace("wannier", relative=True) is layer
    assert blocked.trace(".wannier") is layer

    root.add_strat("other", _strat({"label": "other"}))
    root.grow("other")
    with pytest.raises(ValueError, match="2 children"):
        root.trace(1)


def test_save_writes_one_layer_for_later_evaluation(tmp_path: Path) -> None:
    from qten.utils import io

    io._io_dir = None
    io._all_env = None
    io._current_env = None
    io.iodir(tmp_path)
    io.env("diamond-rg")
    try:
        root = Node.new(name="diamond", operator=_tensor("primitive"))
        root.record("plots", {"band": lambda: None})
        root.add_strat(
            "blocking",
            _strat({"label": "folded", "kind": "blocking", "factor": 3}),
        )
        root.add_strat("wannier", _strat({"label": "courier", "edge": "Wc"}))
        blocked = root.grow("blocking")
        blocked.record("plots", {"band": lambda: None})
        blocked.record("frozen", {"z2": (0, 0, 0, 0), "plots": {"nk": lambda: None}})
        # An unpicklable parent record must not be pulled in with the child.
        root.record("closure", lambda: None)

        version = blocked.save()
        again = blocked.save()
        restored = Node.load("blocking", version=version)
        latest = Node.load("blocking")

        assert version == 1
        assert again == 2
        assert restored.is_root() is True
        assert restored.info["saved_path"] == "blocking"
        assert "plots" not in restored.info
        assert restored.info["frozen"] == {"z2": (0, 0, 0, 0)}
        assert cast(FakeTensor, restored.operator()).name == "folded"
        assert restored.step is not None
        assert restored.step.kind == "blocking"
        assert restored.step.params["factor"] == 3
        assert restored.step.transform is None
        assert cast(FakeTensor, latest.operator()).name == "folded"
        assert [entry["kind"] for entry in root.history] == ["grow", "save", "save"]
        assert root.history[-1]["name"] == "blocking"
        assert root.history[-1]["version"] == 2

        courier = blocked.grow("wannier")
        courier.save()
        loaded = Node.load("blocking.wannier")
        assert loaded.path() == ""
        assert loaded.info["saved_path"] == "blocking.wannier"
        assert cast(FakeTensor, loaded.operator()).name == "courier"
        assert loaded.step is not None
        assert cast(FakeTensor, loaded.step.transform).name == "Wc"

        with pytest.raises(ValueError, match="path segment"):
            root.save("cycle/1")
    finally:
        io._io_dir = None
        io._all_env = None
        io._current_env = None


def test_snapshot_roundtrip_drops_plots_and_restores_the_step() -> None:
    root = Node.new(name="diamond", operator=_tensor("primitive"))
    root.record("plots", {"band": lambda: None})
    root.add_strat("blocking", _strat({"label": "folded", "kind": "blocking", "factor": 3}))
    blocked = root.grow("blocking")
    blocked.record("frozen", {"z2": (0, 0, 0, 0), "plots": {"nk": lambda: None}})

    payload = pickle.loads(pickle.dumps(blocked.snapshot()))
    restored = Node.from_snapshot(payload)

    assert restored.is_root() is True
    assert restored.path() == ""
    assert restored.info["saved_path"] == "blocking"
    assert "plots" not in restored.info
    assert restored.info["frozen"] == {"z2": (0, 0, 0, 0)}
    assert cast(FakeTensor, restored.operator()).name == "folded"
    assert restored.step is not None
    assert restored.step.kind == "blocking"
    assert restored.step.params["factor"] == 3
    assert restored.step.transform is None
    with pytest.raises(TypeError, match="not a qRG node snapshot"):
        Node.from_snapshot({"format": "other"})


def test_save_requires_an_io_environment(tmp_path: Path) -> None:
    from qten.utils import io

    io._io_dir = None
    io._all_env = None
    io._current_env = None
    io.iodir(tmp_path)
    root = Node.new(name="diamond", operator=_tensor("primitive"))
    try:
        with pytest.raises(RuntimeError, match="environment"):
            root.save()
    finally:
        io._io_dir = None
        io._all_env = None
        io._current_env = None


def test_history_records_grows_and_exported_figures(tmp_path: Path) -> None:
    root = Node.new(name="root", operator=_tensor("root"))
    root.add_strat("blocking", _strat({"label": "folded", "kind": "blocking", "factor": 3}))
    blocked = root.grow("blocking")
    figure = tmp_path / "layers" / "blocking.png"
    blocked.plot("tree", export=figure)

    assert figure.is_file()
    assert figure.stat().st_size > 0
    assert [entry["kind"] for entry in root.history] == ["grow", "export"]
    assert root.history[0]["path"] == "blocking"
    assert root.history[0]["depth"] == 1
    assert root.history[0]["impl"] == "unit-pass"
    assert root.history[0]["params"]["factor"] == 3
    assert root.history[1]["file"].endswith("blocking.png")
    assert "time" in root.history[0]


def test_plot_tree_networkx_matplotlib_and_plotly() -> None:
    root = Node.new(name="root node with a very long label", operator=_tensor("root"))
    root.add_strat(
        "child_leaf", _strat({"label": "child", "name": "child node with a very long label"})
    )
    root.add_strat("grandchild_leaf", _strat({"label": "grand"}))
    child = root.grow("child_leaf")
    grandchild = child.grow("grandchild_leaf")

    graph = cast(Any, child.plot("tree", backend="networkx"))
    root_id, child_id, grandchild_id = id(root), id(child), id(grandchild)
    assert set(graph.nodes) == {root_id, child_id, grandchild_id}
    assert graph.edges[root_id, child_id]["target"] == "child_leaf"
    assert graph.graph["positions"][child_id][1] == -1.0

    fig = cast(Any, root.plot("tree", label_width=12))
    labels = [text.get_text() for text in fig.axes[0].texts]
    assert any("\n" in label for label in labels)

    pytest.importorskip("plotly")
    plotly_fig = cast(Any, root.plot("tree", backend="plotly"))
    assert plotly_fig.layout.title.text.startswith("Tree View:")
    assert len(plotly_fig.layout.annotations) == 3


def test_grow_on_error_record_attaches_error_node() -> None:
    def broken(node: Node, **_params: Any) -> Branch:
        raise ValueError("singular value dropped to zero")

    register("broken-pass", broken)
    root = Node.new(name="root", operator=_tensor("root"))
    try:
        root.add_strat("wannier", Strategy("broken-pass", {"seed": "bad"}))
        with pytest.raises(ValueError, match="singular value"):
            root.grow("wannier")
        failed_child = root.grow("wannier", on_error="record")
        assert failed_child.failed is True
        assert failed_child.path() == "wannier"
        assert failed_child.step is not None
        assert failed_child.step.kind == "error"
        assert failed_child.info["error"]["type"] == "ValueError"
        assert "singular value" in failed_child.info["error"]["message"]
        with pytest.raises(RuntimeError, match="failed with ValueError"):
            failed_child.operator()
        assert root.history[-1]["kind"] == "error"
        fig = cast(Any, failed_child.plot("tree"))
        labels = [text.get_text() for text in fig.axes[0].texts]
        assert any("ERROR: wannier" in label for label in labels)
    finally:
        unregister("broken-pass")
