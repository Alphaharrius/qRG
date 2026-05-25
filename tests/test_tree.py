import pickle
from pathlib import Path
from typing import Any, cast

import pytest

import qrg.tree as tree_module
from qrg.tree import Node


class FakeTensor:
    def __init__(
        self,
        name: str,
        dims: tuple[str, str] = ("d0", "d1"),
        device: str = "cpu",
    ) -> None:
        self.name = name
        self.dims = dims
        self.device = device

    def __matmul__(self, other: "FakeTensor") -> "FakeTensor":
        return FakeTensor(f"({self.name}@{other.name})", dims=self.dims, device=self.device)

    def h(self, *_axes: int) -> "FakeTensor":
        return FakeTensor(f"{self.name}.h", dims=self.dims, device=self.device)


def _tensor_stub() -> Any:
    return cast(Any, object())


def test_grow_sets_parent_and_uses_child_target_for_transform() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"external_target_name": child}

    root = Node.new(name="root", target=root_tensor, methods={"grow": grow_method})
    root.grow("grow")

    leaf = root.leaf("external_target_name")
    assert leaf.node.parent() is root
    assert leaf.transform is child_tensor


def test_child_grow_looks_up_methods_from_root() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()
    grandchild_tensor = _tensor_stub()

    def root_grow(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    def child_grow(_: Node) -> dict[str, Node]:
        grandchild = Node(
            name="grandchild",
            _target="grandchild_transform",
            _data={"grandchild_transform": grandchild_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"grandchild": grandchild}

    root = Node.new(
        name="root",
        target=root_tensor,
        methods={"root_grow": root_grow, "child_grow": child_grow},
    )
    root.grow("root_grow")

    child = root.leaf("child").node
    child.grow("child_grow")
    grandchild = child.leaf("grandchild")
    assert grandchild.node.parent() is child
    assert grandchild.transform is grandchild_tensor


def test_register_method_from_child_updates_root_registry() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    def new_method(_: Node) -> dict[str, Node]:
        return {}

    root = Node.new(name="root", target=root_tensor, methods={"grow": grow_method})
    root.grow("grow")

    child = root.leaf("child").node
    child.register_method("new", new_method)
    assert root._methods["new"] is new_method
    assert "new" not in child._methods


def test_target_returns_tensor_at_current_target_key() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(name="root", target=root_tensor, methods={"grow": grow_method})
    root.grow("grow")

    child = root.leaf("child").node
    assert root.target() is root_tensor
    assert child.target() is child_tensor


def test_compute_stores_derived_tensor_in_data() -> None:
    root_tensor = _tensor_stub()
    derived_tensor = _tensor_stub()

    root = Node.new(name="root", target=root_tensor, methods={})
    returned = root.compute("derived", lambda node: derived_tensor)

    assert returned is root
    assert root["derived"] is derived_tensor


def test_cut_detaches_direct_branch_into_new_root() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(name="root", target=root_tensor, methods={"grow": grow_method})
    root.grow("grow")

    child = root.cut("child")

    assert child.name == "child"
    assert child.root() is child
    with pytest.raises(ValueError, match="Target child not found in leaves of node root"):
        root.leaf("child")
    with pytest.raises(ValueError, match="Node child has no parent"):
        child.parent()


def test_cut_raises_for_missing_direct_branch() -> None:
    root = Node.new(name="root", target=_tensor_stub(), methods={})

    with pytest.raises(ValueError, match="Target missing not found in leaves of node root"):
        root.cut("missing")


def test_branches_returns_shallow_copy_of_direct_leaves() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(name="root", target=root_tensor, methods={"grow": grow_method})
    root.grow("grow")

    branches = root.branches()

    assert set(branches) == {"child"}
    assert branches["child"].node is root.leaf("child").node

    branches.pop("child")
    assert set(root.branches()) == {"child"}


def test_root_returns_topmost_ancestor() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(name="root", target=root_tensor, methods={"grow": grow_method})
    root.grow("grow")

    child = root.leaf("child").node
    assert root.root() is root
    assert child.root() is root


def test_is_root_reflects_parent_link() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(name="root", target=root_tensor, methods={"grow": grow_method})
    root.grow("grow")
    child = root.leaf("child").node

    assert root.is_root() is True
    assert child.is_root() is False


def test_is_leaf_reflects_whether_node_has_direct_branches() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(name="root", target=root_tensor, methods={"grow": grow_method})

    assert root.is_leaf() is True

    root.grow("grow")
    child = root.leaf("child").node

    assert root.is_leaf() is False
    assert child.is_leaf() is True


def test_trace_resolves_absolute_and_relative_paths() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()
    grandchild_tensor = _tensor_stub()

    def child_grow(_: Node) -> dict[str, Node]:
        grandchild = Node(
            name="grandchild",
            _target="grandchild_transform",
            _data={"grandchild_transform": grandchild_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"grandchild": grandchild}

    def root_grow(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(
        name="root",
        target=root_tensor,
        methods={"root_grow": root_grow, "child_grow": child_grow},
    )
    root.grow("root_grow")
    child = root.leaf("child").node
    child.grow("child_grow")
    grandchild = child.leaf("grandchild").node

    assert root.trace("") is root
    assert child.trace("") is root
    assert child.trace(".") is child
    assert root.trace("child.grandchild") is grandchild
    assert child.trace(".grandchild") is grandchild


def test_path_returns_absolute_leaf_key_trace() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()
    grandchild_tensor = _tensor_stub()

    def child_grow(_: Node) -> dict[str, Node]:
        grandchild = Node(
            name="grandchild",
            _target="grandchild_transform",
            _data={"grandchild_transform": grandchild_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"grandchild": grandchild}

    def root_grow(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(
        name="root",
        target=root_tensor,
        methods={"root_grow": root_grow, "child_grow": child_grow},
    )
    root.grow("root_grow")
    child = root.leaf("child").node
    child.grow("child_grow")
    grandchild = child.leaf("grandchild").node

    assert root.path() == ""
    assert child.path() == "child"
    assert grandchild.path() == "child.grandchild"
    assert root.trace(grandchild.path()) is grandchild


def test_trace_raises_with_explicit_break_position() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def root_grow(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(name="root", target=root_tensor, methods={"grow": root_grow})
    root.grow("grow")
    child = root.leaf("child").node

    with pytest.raises(ValueError, match=r'trace cannot be resolve at "child\.\[missing\]\.leaf"'):
        root.trace("child.missing.leaf")
    with pytest.raises(ValueError, match=r'trace cannot be resolve at "\[missing\]\.leaf"'):
        child.trace(".missing.leaf")
    with pytest.raises(ValueError, match='Invalid trace expression "\\.child\\."'):
        root.trace(".child.")


def test_find_filters_current_subtree_by_regex_and_predicate() -> None:
    root_tensor = _tensor_stub()
    alpha_tensor = _tensor_stub()
    beta_tensor = _tensor_stub()
    gamma_tensor = _tensor_stub()

    def gamma_grow(_: Node) -> dict[str, Node]:
        gamma = Node(
            name="gamma-child",
            _target="gamma_transform",
            _data={"gamma_transform": gamma_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"gamma": gamma}

    def root_grow(_: Node) -> dict[str, Node]:
        alpha = Node(
            name="alpha-child",
            _target="alpha_transform",
            _data={"alpha_transform": alpha_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        beta = Node(
            name="beta-child",
            _target="beta_transform",
            _data={"beta_transform": beta_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"alpha": alpha, "beta": beta}

    root = Node.new(
        name="root",
        target=root_tensor,
        methods={"root_grow": root_grow, "gamma_grow": gamma_grow},
    )
    root.grow("root_grow")
    alpha = root.leaf("alpha").node
    beta = root.leaf("beta").node
    alpha.grow("gamma_grow")
    gamma = alpha.leaf("gamma").node

    assert root.find(regex="child$") == [alpha, gamma, beta]
    assert root.find(predicate=lambda node: node is not root) == [alpha, gamma, beta]
    assert root.find(regex="^a", predicate=lambda node: "alpha" in node.name) == [alpha]
    assert alpha.find(regex="child$") == [alpha, gamma]


def test_find_requires_at_least_one_filter() -> None:
    root = Node.new(name="root", target=_tensor_stub(), methods={})

    with pytest.raises(ValueError, match="find\\(\\) requires regex and/or predicate"):
        root.find()


def test_get_transform_returns_ordered_list_and_composed_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_tensor: Any = FakeTensor("root_target")
    child_transform: Any = FakeTensor("child_transform")
    grandchild_transform: Any = FakeTensor("grandchild_transform")

    def child_grow(_: Node) -> dict[str, Node]:
        grandchild = Node(
            name="grandchild",
            _target="grandchild_transform",
            _data={"grandchild_transform": grandchild_transform},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"grandchild": grandchild}

    def root_grow(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_transform},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(
        name="root",
        target=root_tensor,
        methods={"root_grow": root_grow, "child_grow": child_grow},
    )
    root.grow("root_grow")
    child = root.leaf("child").node
    child.grow("child_grow")

    monkeypatch.setattr(
        cast(Any, tree_module.qten),  # type: ignore[attr-defined]
        "eye",
        lambda dims, *, device=None: FakeTensor("identity", dims=dims, device=device),
    )

    assert root.get_transform("child.grandchild", composed=False) == [
        child_transform,
        grandchild_transform,
    ]
    assert cast(Any, root.get_transform("child.grandchild")).name == (
        "(child_transform@grandchild_transform)"
    )
    assert child.get_transform(".grandchild", composed=False) == [grandchild_transform]
    assert cast(Any, child.get_transform(".")).name == "identity"


def test_get_transform_rejects_non_descendant_trace() -> None:
    root_tensor: Any = FakeTensor("root_target")
    left_transform: Any = FakeTensor("left_transform")
    right_transform: Any = FakeTensor("right_transform")

    def root_grow(_: Node) -> dict[str, Node]:
        left = Node(
            name="left",
            _target="left_transform",
            _data={"left_transform": left_transform},
            _parent=None,
            _leaves={},
            _methods={},
        )
        right = Node(
            name="right",
            _target="right_transform",
            _data={"right_transform": right_transform},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"left": left, "right": right}

    root = Node.new(name="root", target=root_tensor, methods={"grow": root_grow})
    root.grow("grow")
    left = root.leaf("left").node

    with pytest.raises(
        ValueError,
        match='Trace "right" does not resolve to the current node or its descendants',
    ):
        left.get_transform("right")


def test_pickle_roundtrip_excludes_methods_registry() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child": child}

    root = Node.new(name="root", target=root_tensor, methods={"grow": grow_method})
    root.grow("grow")

    restored = pickle.loads(pickle.dumps(root))
    restored_child = restored.leaf("child").node

    assert restored._methods == {}
    assert restored_child._methods == {}
    assert restored_child.parent() is restored


def test_register_methods_loads_only_annotated_growths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "growths.py"
    script_path.write_text(
        "\n".join(
            [
                "from qrg.tree import Node",
                "",
                '@Node.growth("loaded")',
                "def loaded_method(node: Node) -> dict[str, Node]:",
                "    return {}",
                "",
                "def ignored_method(node: Node) -> dict[str, Node]:",
                "    return {}",
            ]
        )
    )

    monkeypatch.chdir(tmp_path)
    root = Node.new(name="root", target=_tensor_stub(), methods={})
    root.register_methods("growths.py")

    assert "loaded" in root._methods
    assert root._methods["loaded"].__name__ == "loaded_method"
    assert "ignored_method" not in root._methods


def test_plot_tree_networkx_returns_whole_tree_with_current_highlight() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()
    grandchild_tensor = _tensor_stub()

    def child_grow(_: Node) -> dict[str, Node]:
        grandchild = Node(
            name="grandchild",
            _target="grandchild_transform",
            _data={"grandchild_transform": grandchild_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"grandchild_leaf": grandchild}

    def root_grow(_: Node) -> dict[str, Node]:
        child = Node(
            name="child",
            _target="child_transform",
            _data={"child_transform": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child_leaf": child}

    root = Node.new(
        name="root",
        target=root_tensor,
        methods={"root_grow": root_grow, "child_grow": child_grow},
    )
    root.grow("root_grow")
    child = root.leaf("child_leaf").node
    child.grow("child_grow")
    grandchild = child.leaf("grandchild_leaf").node

    graph = cast(Any, child.plot("tree", backend="networkx"))
    root_id = id(root)
    child_id = id(child)
    grandchild_id = id(grandchild)

    assert set(graph.nodes) == {root_id, child_id, grandchild_id}
    assert set(graph.edges) == {(root_id, child_id), (child_id, grandchild_id)}
    assert graph.graph["root"] is root
    assert graph.graph["current"] is child
    assert graph.nodes[root_id]["node"] is root
    assert graph.nodes[root_id]["is_root"] is True
    assert graph.nodes[root_id]["is_current"] is False
    assert graph.nodes[child_id]["is_current"] is True
    assert graph.edges[root_id, child_id]["target"] == "child_leaf"
    assert graph.edges[child_id, grandchild_id]["target"] == "grandchild_leaf"
    assert graph.graph["positions"][root_id][1] == 0.0
    assert graph.graph["positions"][child_id][1] == -1.0


def test_plot_tree_matplotlib_wraps_labels() -> None:
    root_tensor = _tensor_stub()
    child_tensor = _tensor_stub()

    def grow_method(_: Node) -> dict[str, Node]:
        child = Node(
            name="child node with a very long label",
            _target="a_very_long_target_key_name",
            _data={"a_very_long_target_key_name": child_tensor},
            _parent=None,
            _leaves={},
            _methods={},
        )
        return {"child_leaf": child}

    root = Node.new(
        name="root node with a very long label",
        target=root_tensor,
        methods={"grow": grow_method},
    )
    root.grow("grow")

    fig = cast(Any, root.plot("tree", backend="matplotlib", label_width=12))
    labels = [text.get_text() for text in fig.axes[0].texts]

    assert any("\n" in label for label in labels)


def test_plot_tree_defaults_to_matplotlib_figure() -> None:
    root = Node.new(name="root", target=_tensor_stub(), methods={})

    fig = cast(Any, root.plot("tree"))

    assert hasattr(fig, "axes")


def test_plot_tree_plotly_returns_plotly_figure() -> None:
    pytest.importorskip("plotly")

    root = Node.new(name="root", target=_tensor_stub(), methods={})

    fig = cast(Any, root.plot("tree", backend="plotly"))

    assert hasattr(fig, "to_plotly_json")
    assert fig.layout.title.text == "Tree View: root"
    assert len(fig.layout.annotations) == 1
