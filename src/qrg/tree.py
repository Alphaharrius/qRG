"""
Tree-building helpers for renormalization workflows.

This module provides a lightweight `Node` object that stores named
`qten.Tensor` instances together with expansion methods for growing a tree of
derived nodes. Each node behaves like a mapping from string keys to tensors,
and each growth step records a `_LeavePath` describing both the child node and
the transform tensor that connects the current node's target tensor to the
child target tensor.

The intended workflow is:

1. Create a root node with [`Node.new`][qrg.tree.Node.new].
2. Register one or more growth methods that expand a node into child nodes.
3. Call [`Node.grow`][qrg.tree.Node.grow] to materialize leaf paths.
4. Traverse children through [`Node.leaf`][qrg.tree.Node.leaf] and
   [`Node.parent`][qrg.tree.Node.parent].

Notes
-----
The public growth-method contract is intentionally simple: a growth method
returns `dict[str, Node]`. The returned dictionary key is used as the lookup
key for the new leaf under the current node, while the child node's private
`_target` field determines which tensor inside the child becomes the leaf-path
transform.

Examples
--------
Create and grow a minimal tree:

>>> import qten
>>> from qrg.tree import Node
>>> root_tensor = qten.eye(2)
>>> child_tensor = qten.eye(2)
>>> def grow_once(node: Node) -> dict[str, Node]:
...     return {
...         "child": Node(
...             name="child",
...             _target="iso",
...             _data={"iso": child_tensor},
...             _parent=None,
...             _leaves={},
...             _methods={},
...         )
...     }
>>> root = Node.new(name="root", target=root_tensor, methods={"step": grow_once})
>>> root.grow("step")
Node(...)
>>> path = root.leaf("child")
>>> path.node.parent() is root
True
"""

import importlib.util
import re
import textwrap
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]
import qten
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

TensorLike = qten.Tensor[Any]


@dataclass
class _LeavePath:
    """
    Leaf metadata stored on a parent node after growth.

    Parameters
    ----------
    node : Node
        Child node reached from the parent node.
    transform : qten.Tensor
        Tensor that maps the parent node's target tensor into the child node's
        target tensor basis.

    Notes
    -----
    The expected interpretation is
    `transform.h(-2, -1) @ target @ transform`, where `target` is the parent
    node's target tensor. This class does not perform that contraction itself;
    it only stores the bookkeeping needed to do so elsewhere.

    Examples
    --------
    A `_LeavePath` is usually created by [`Node.grow`][qrg.tree.Node.grow]
    rather than instantiated directly.
    """

    node: "Node"
    transform: TensorLike


_GrowMethod = Callable[["Node"], dict[str, "Node"]]
_GROWTH_MARKER = "_qrg_growth_name"


@dataclass
class Node(Mapping[str, TensorLike], qten.plottings.Plottable):
    """
    Tree node containing tensors, growth methods, and derived leaf paths.

    `Node` serves two roles:

    - A mapping from string names to `qten.Tensor` objects.
    - A tree vertex that can spawn children through registered growth methods.

    Parameters
    ----------
    name : str
        Human-readable identifier for the node.
    _target : str
        Key in `_data` identifying the node's target tensor. This key is used
        by the parent node when extracting the transform tensor for a leaf path.
    _data : dict[str, qten.Tensor]
        Tensor payload owned by the node.
    _parent : Node or None
        Parent node in the tree. The root node stores `None`.
    _leaves : dict[str, _LeavePath]
        Cached leaf paths produced by growth methods.
    _methods : dict[str, callable]
        Registered growth methods. Each method receives the current node and
        returns a mapping from leaf lookup names to child nodes.

    Notes
    -----
    `Node` inherits from `Mapping`, so `node[key]`, `key in node`, iteration,
    and `len(node)` are delegated to `_data`.

    Growth methods are owned by the root node. Registering a method from any
    node forwards the registration to the root, and growth-method lookup also
    resolves through the root only.

    The tree API distinguishes between:

    - The leaf lookup key in `_leaves`, chosen by the growth method's returned
      dictionary key.
    - The child node's `_target`, which identifies the tensor inside the child
      that becomes the stored `_LeavePath.transform`.

    When a node is pickled, `_methods` is excluded from the serialized state.
    After unpickling, the method registry is restored as an empty dictionary
    and must be repopulated explicitly by the caller.

    Through [`plot()`][qten.plottings.Plottable.plot], nodes also support a
    `tree` plot method with backends `networkx`, `matplotlib`, and `plotly`.
    The `networkx` backend returns a directed graph for the whole tree rooted
    at `node.root()`, with the current node highlighted in node attributes and
    layout positions stored in graph-level metadata. The `matplotlib` and
    `plotly` backends render that tree with boxed, wrapped labels and
    highlight the current node directly in the figure. Calling
    `node.plot("tree")` without an explicit backend defaults to the
    `matplotlib` tree renderer.

    Examples
    --------
    Build a root node and access its target tensor by mapping key:

    >>> import qten
    >>> root = Node.new(name="root", target=qten.eye(2), methods={})
    >>> "root" in root
    True
    >>> root["root"]
    Tensor(...)
    """

    name: str
    _target: str
    _data: dict[str, TensorLike]
    _parent: "Node | None"
    _leaves: dict[str, _LeavePath]
    _methods: dict[str, _GrowMethod]

    def __getitem__(self, key: str) -> TensorLike:
        """
        Return a tensor stored on this node by name.

        Parameters
        ----------
        key : str
            Tensor name to retrieve from the node payload.

        Returns
        -------
        qten.Tensor
            Stored tensor associated with `key`.

        Raises
        ------
        KeyError
            If `key` is not present in `_data`.
        """
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over stored tensor names.

        Returns
        -------
        iterator of str
            Iterator over `_data` keys in insertion order.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """
        Return the number of tensors stored on this node.

        Returns
        -------
        int
            Number of entries in `_data`.
        """
        return len(self._data)

    def plot(
        self,
        method: str,
        backend: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> object:
        """
        Dispatch plotting calls, defaulting tree plots to matplotlib output.

        Parameters
        ----------
        method : str
            Plot method name.
        backend : str or None, optional
            Backend name requested by the caller. When omitted, tree plots
            default to `matplotlib` while other plot methods keep the usual
            `plotly` default.
        args
            Positional arguments forwarded to the selected backend.
        kwargs
            Keyword arguments forwarded to the selected backend.

        Returns
        -------
        object
            Backend-specific plot object returned by the selected renderer.
        """
        if backend is None or backend == "":
            backend = "matplotlib" if method == "tree" else "plotly"
        return super().plot(method, backend, *args, **kwargs)

    def __getstate__(self) -> dict[str, object]:
        """
        Return pickle state for the node without the method registry.

        Returns
        -------
        dict
            Instance state with `_methods` replaced by an empty dictionary.

        Notes
        -----
        Excluding `_methods` keeps serialization independent of whether the
        registered callables are picklable. Callers are expected to restore the
        root method registry manually after loading.
        """
        state = cast(dict[str, object], self.__dict__.copy())
        state["_methods"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """
        Restore node state from pickle data.

        Parameters
        ----------
        state : dict
            Previously pickled instance state.

        Notes
        -----
        `_methods` is always restored to an empty dictionary if it is missing
        or was intentionally stripped during pickling.
        """
        self.__dict__.update(state)
        self._methods = {}

    def register_method(self, method_name: str, method: _GrowMethod) -> Self:
        """
        Register or replace a growth method on the root registry.

        Parameters
        ----------
        method_name : str
            Name used later with [`grow()`][qrg.tree.Node.grow].
        method : callable
            Callable receiving this node and returning
            `dict[str, Node]`. The returned dictionary keys become leaf lookup
            names under this node.

        Returns
        -------
        Node
            The current node, allowing fluent chaining even when registration
            is initiated from a descendant.

        Notes
        -----
        Registration is always applied to the root node's `_methods`
        dictionary. Registering a method with an existing name replaces the
        previous method without warning.

        Examples
        --------
        >>> root = Node.new(name="root", target=qten.eye(2), methods={})
        >>> root.register_method("step", lambda node: {})
        Node(...)
        """
        self.root()._methods[method_name] = method
        return self

    @staticmethod
    def growth(name: str) -> Callable[[_GrowMethod], _GrowMethod]:
        """
        Mark a function in an external script as a loadable growth method.

        Parameters
        ----------
        name : str
            Registry name to use when the function is discovered by
            [`register_methods()`][qrg.tree.Node.register_methods].

        Returns
        -------
        callable
            Decorator that attaches growth-registration metadata to the
            function and returns it unchanged.

        Notes
        -----
        This decorator does not register the function immediately. It only
        marks the function so that [`register_methods()`][qrg.tree.Node.register_methods]
        can discover it when loading a script file.

        Examples
        --------
        >>> @Node.growth("step")
        ... def grow_step(node: Node) -> dict[str, Node]:
        ...     return {}
        """

        def decorator(method: _GrowMethod) -> _GrowMethod:
            setattr(method, _GROWTH_MARKER, name)
            return method

        return decorator

    def register_methods(self, path: str) -> Self:
        """
        Load and register annotated growth methods from a Python script.

        Parameters
        ----------
        path : str
            Path to a Python source file. Relative paths are resolved against
            the current working directory.

        Returns
        -------
        Node
            The current node, allowing fluent chaining even when registration
            is initiated from a descendant.

        Raises
        ------
        FileNotFoundError
            If `path` does not point to an existing file.
        ImportError
            If the script cannot be loaded as a Python module.
        Exception
            Propagates any exception raised while executing the script.

        Notes
        -----
        Only callables marked with [`@Node.growth(...)`][qrg.tree.Node.growth]
        are registered. Unannotated functions in the script are ignored.

        Registration is forwarded to the root node's registry in the same way
        as [`register_method()`][qrg.tree.Node.register_method].

        Examples
        --------
        >>> root = Node.new(name="root", target=qten.eye(2), methods={})
        >>> root.register_methods("growths.py")
        Node(...)
        """
        script_path = Path(path)
        if not script_path.is_absolute():
            script_path = Path.cwd() / script_path
        script_path = script_path.resolve()

        if not script_path.is_file():
            raise FileNotFoundError(f"Growth-method script not found: {script_path}")

        module_name = f"_qrg_growths_{script_path.stem}_{abs(hash(script_path))}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load growth-method script: {script_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for value in vars(module).values():
            growth_name = getattr(value, _GROWTH_MARKER, None)
            if growth_name is None:
                continue
            self.register_method(growth_name, value)
        return self

    def target(self) -> TensorLike:
        """
        Return the current target tensor for this node.

        Returns
        -------
        qten.Tensor
            Tensor stored at the node's `_target` key.

        Raises
        ------
        KeyError
            If `_target` is not present in `_data`.

        Notes
        -----
        This is a convenience accessor for `self[self._target]`.

        Examples
        --------
        >>> root = Node.new(name="root", target=qten.eye(2), methods={})
        >>> root.target()
        Tensor(...)
        """
        return self[self._target]

    def compute(self, name: str, f: Callable[["Node"], TensorLike]) -> Self:
        """
        Compute and store a derived tensor on this node.

        Parameters
        ----------
        name : str
            Key under which to store the computed tensor in `_data`.
        f : callable
            Callable receiving the current node and returning the tensor to
            store.

        Returns
        -------
        Node
            The current node, allowing fluent chaining.

        Raises
        ------
        Exception
            Propagates any exception raised by `f(node)`.

        Notes
        -----
        If `name` already exists in `_data`, its previous value is replaced.

        Examples
        --------
        >>> root = Node.new(name="root", target=qten.eye(2), methods={})
        >>> root.compute("copy", lambda node: node.target())
        Node(...)
        >>> root["copy"]
        Tensor(...)
        """
        self._data[name] = f(self)
        return self

    def leaf(self, target: str) -> _LeavePath:
        """
        Return a previously materialized leaf path by lookup key.

        Parameters
        ----------
        target : str
            Leaf lookup key produced by a prior call to
            [`grow()`][qrg.tree.Node.grow].

        Returns
        -------
        _LeavePath
            Stored path containing the child node and its transform tensor.

        Raises
        ------
        ValueError
            If `target` is not present in `_leaves`. This usually means the
            node has not been grown with the relevant method yet, or the
            requested leaf name does not exist.

        Examples
        --------
        After `grow()` completes, retrieve one child path:

        >>> path = root.leaf("child")
        >>> path.node
        Node(...)
        """
        if target not in self._leaves:
            raise ValueError(f"Target {target} not found in leaves of node {self.name}")
        return self._leaves[target]

    def branches(self) -> dict[str, _LeavePath]:
        """
        Return a shallow copy of the current node's direct branches.

        Returns
        -------
        dict[str, _LeavePath]
            Mapping from direct leaf lookup keys to stored leaf-path metadata.

        Notes
        -----
        The returned dictionary is a shallow copy, so mutating it does not
        modify the node's internal `_leaves` mapping.
        """
        return dict(self._leaves)

    def cut(self, name: str) -> "Node":
        """
        Detach a direct child branch from the current node.

        Parameters
        ----------
        name : str
            Leaf lookup key under the current node identifying the branch to
            detach.

        Returns
        -------
        Node
            Root node of the detached subtree.

        Raises
        ------
        ValueError
            If `name` is not present in the current node's leaves.

        Notes
        -----
        The returned node becomes the root of its own tree by clearing its
        parent link. Only the direct branch selected by `name` is removed from
        the current node.
        """
        if name not in self._leaves:
            raise ValueError(f"Target {name} not found in leaves of node {self.name}")
        leaf = self._leaves.pop(name)
        leaf.node._parent = None
        return leaf.node

    def trace(self, expr: str) -> "Node":
        """
        Resolve a node by dot-delimited leaf path.

        Parameters
        ----------
        expr : str
            Trace expression to resolve. Expressions of the form
            `"a.b.c"` are resolved from the root node. Expressions beginning
            with `"."`, such as `".a.b"`, are resolved relative to the
            current node. The empty absolute path `""` resolves to the root,
            and `"."` resolves to the current node.

        Returns
        -------
        Node
            Node reached by following the requested leaf path.

        Raises
        ------
        ValueError
            If the trace expression is malformed or cannot be fully resolved.
            Errors identify the segment where resolution failed as
            `trace cannot be resolve at "a.[b].c"`.
        """
        is_relative = expr.startswith(".")
        raw_path = expr[1:] if is_relative else expr
        current = self if is_relative else self.root()

        if raw_path == "":
            return current

        parts = raw_path.split(".")
        if not parts or any(part == "" for part in parts):
            raise ValueError(f'Invalid trace expression "{expr}"')
        for index, part in enumerate(parts):
            if part not in current._leaves:
                resolved = parts[:index]
                unresolved = parts[index + 1 :]
                highlighted = ".".join(resolved + [f"[{part}]"] + unresolved)
                raise ValueError(f'trace cannot be resolve at "{highlighted}"')
            current = current._leaves[part].node
        return current

    def path(self) -> str:
        """
        Return the dot-delimited absolute leaf-key path from the root.

        Returns
        -------
        str
            Canonical absolute path from the root to this node. The root
            node's path is the empty string `""`.

        Raises
        ------
        ValueError
            If the node is not reachable from its recorded parent chain using
            the parents' leaf mappings.
        """
        if self._parent is None:
            return ""

        parts: list[str] = []
        current = self
        while current._parent is not None:
            parent = current._parent
            for leaf_name, leaf in parent._leaves.items():
                if leaf.node is current:
                    parts.append(leaf_name)
                    current = parent
                    break
            else:
                raise RuntimeError(
                    f"Node {current.name} is not reachable from parent {parent.name}"
                )
        parts.reverse()
        return ".".join(parts)

    def get_transform(self, expr: str, *, composed: bool = True) -> TensorLike | list[TensorLike]:
        """
        Return the transform(s) from the current node to a traced descendant.

        Parameters
        ----------
        expr : str
            Trace expression interpreted with the same rules as
            [`trace()`][qrg.tree.Node.trace].
        composed : bool, default=True
            If `True`, return a single composed transform tensor `T`. If
            `False`, return the ordered list of direct branch transforms whose
            product would form `T`.

        Returns
        -------
        qten.Tensor or list[qten.Tensor]
            Either the composed transform or the ordered list of transforms
            from the current node to the traced node.

        Raises
        ------
        ValueError
            If `expr` resolves to a node outside the current node's subtree.

        Notes
        -----
        For a descendant reached by transforms `[T1, T2, ..., Tn]`, the
        composed result is `T1 @ T2 @ ... @ Tn`, so the descendant target is
        obtained from the current target as
        `T.h(-2, -1) @ target @ T`.

        Using this transform chain in the inverse/backward direction may
        create very large intermediate tensors. In particular, for structured
        tensor types such as momentum-block tensors, lifting a descendant
        operator back toward an ancestor can expand into a large pair-resolved
        space. For actual transport computations, staged application with
        `composed=False` is usually preferable to materializing one large
        composed transform.
        """
        destination = self.trace(expr)
        source_path = self.path()
        destination_path = destination.path()

        if source_path == destination_path:
            relative_parts: list[str] = []
        elif source_path == "":
            relative_parts = destination_path.split(".")
        elif destination_path.startswith(source_path + "."):
            relative_parts = destination_path[len(source_path) + 1 :].split(".")
        else:
            raise ValueError(
                f'Trace "{expr}" does not resolve to the current node or its descendants'
            )

        transforms: list[TensorLike] = []
        current = self
        for part in relative_parts:
            leaf = current.leaf(part)
            transforms.append(leaf.transform)
            current = leaf.node

        if not composed:
            return transforms

        if not transforms:
            target = self.target()
            return qten.eye(target.dims, device=target.device)

        transform = transforms[0]
        for step in transforms[1:]:
            transform = transform @ step
        return transform

    def find(
        self, *, regex: str | None = None, predicate: Callable[["Node"], bool] | None = None
    ) -> list["Node"]:
        """
        Find nodes in the current subtree by name regex and/or predicate.

        Parameters
        ----------
        regex : str, optional
            Regular expression matched against each node's `name`.
        predicate : callable, optional
            Additional boolean filter applied to each visited node.

        Returns
        -------
        list[Node]
            All matching nodes in depth-first traversal order, starting from
            the current node.

        Raises
        ------
        ValueError
            If neither `regex` nor `predicate` is provided.

        Notes
        -----
        When both filters are supplied, a node must satisfy both.
        """
        if regex is None and predicate is None:
            raise ValueError("find() requires regex and/or predicate")

        pattern = re.compile(regex) if regex is not None else None
        matches: list[Node] = []
        stack = [self]

        while stack:
            current = stack.pop()
            regex_match = pattern.search(current.name) is not None if pattern else True
            predicate_match = predicate(current) if predicate else True
            if regex_match and predicate_match:
                matches.append(current)

            children = [path.node for path in current._leaves.values()]
            stack.extend(reversed(children))

        return matches

    def parent(self) -> "Node":
        """
        Return the parent node.

        Returns
        -------
        Node
            Parent of the current node.

        Raises
        ------
        ValueError
            If this node is the root and therefore has no parent.

        Notes
        -----
        Child parent links are assigned automatically by
        [`grow()`][qrg.tree.Node.grow].
        """
        if self._parent is None:
            raise ValueError(f"Node {self.name} has no parent")
        return self._parent

    def root(self) -> "Node":
        """
        Return the root node of the current tree.

        Returns
        -------
        Node
            Top-most ancestor reachable by following parent links. For the root
            node itself, this method returns `self`.

        Notes
        -----
        The root owns the tree-wide growth-method registry used by
        [`register_method()`][qrg.tree.Node.register_method] and
        [`grow()`][qrg.tree.Node.grow].

        Examples
        --------
        >>> root = Node.new(name="root", target=qten.eye(2), methods={})
        >>> root.root() is root
        True
        """
        root = self
        while root._parent is not None:
            root = root._parent
        return root

    def is_root(self) -> bool:
        """
        Return whether this node is the root of its tree.

        Returns
        -------
        bool
            `True` when the node has no parent, otherwise `False`.
        """
        return self._parent is None

    def is_leaf(self) -> bool:
        """
        Return whether this node currently has no child branches.

        Returns
        -------
        bool
            `True` when the node has no direct leaves, otherwise `False`.
        """
        return len(self._leaves) == 0

    def grow(self, method: str) -> Self:
        """
        Expand this node using a registered growth method.

        Parameters
        ----------
        method : str
            Name of a previously registered growth method.

        Returns
        -------
        Node
            The current node after updating `_leaves`.

        Raises
        ------
        ValueError
            If `method` has not been registered on this node.
        KeyError
            If a returned child node does not contain its `_target` key in its
            `_data` mapping. In that case the leaf transform cannot be derived.

        Notes
        -----
        The growth method returns `dict[str, Node]`. For each returned child:

        - The dictionary key becomes the lookup key in `self._leaves`.
        - The transform stored in `_LeavePath` is `child[child._target]`.
        - The child node's `_parent` is set to `self`.
        - The growth method itself is always resolved from the root node's
          `_methods` registry, regardless of which node calls `grow()`.

        If a leaf key already exists, it is overwritten by the latest growth
        result with the same name.

        Examples
        --------
        >>> root.grow("step")
        Node(...)
        >>> child_path = root.leaf("child")
        >>> child_path.node.parent() is root
        True
        """
        root = self.root()
        if method not in root._methods:
            raise ValueError(f"Method {method} not found in node {self.name}")
        new_nodes = root._methods[method](self)
        new_leaves = {
            target: _LeavePath(node=node, transform=node[node._target])
            for target, node in new_nodes.items()
        }
        for leaf in new_leaves.values():
            leaf.node._parent = self
        self._leaves.update(new_leaves)
        return self

    @staticmethod
    def new(*, name: str, target: TensorLike, methods: dict[str, _GrowMethod]) -> "Node":
        """
        Construct a root node from a single target tensor.

        Parameters
        ----------
        name : str
            Root node name. This also becomes the initial `_target` key.
        target : qten.Tensor
            Target tensor stored on the root node under `name`.
        methods : dict[str, callable]
            Initial growth-method registry for the root node.

        Returns
        -------
        Node
            Newly created root node with no parent and no leaves.

        Notes
        -----
        The created node stores:

        - `_target == name`
        - `_data == {name: target}`
        - `_parent is None`

        Examples
        --------
        >>> import qten
        >>> root = Node.new(name="hamiltonian", target=qten.eye(2), methods={})
        >>> root.name
        'hamiltonian'
        >>> list(root)
        ['hamiltonian']
        """
        return Node(
            name=name,
            _target=name,
            _data={name: target},
            _parent=None,
            _leaves={},
            _methods=methods,
        )


def _tree_layout(root: Node) -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    next_x = 0.0

    def assign(node: Node, depth: int) -> float:
        nonlocal next_x
        children = [path.node for path in node._leaves.values()]
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


def _wrap_node_label(label: str, target_key: str, *, width: int) -> str:
    text = f"{label}\n[{target_key}]"
    return "\n".join(textwrap.fill(line, width=width) for line in text.splitlines())


def _tree_node_style(*, is_current: bool, is_root: bool) -> dict[str, object]:
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
    Build a `networkx` graph representation of the full node tree.

    Parameters
    ----------
    node : Node
        Current node from which plotting was requested.

    Returns
    -------
    networkx.DiGraph
        Directed graph containing every node reachable from `node.root()`.
        Graph metadata includes:

        - `graph["root"]`: root node of the tree
        - `graph["current"]`: node on which `plot()` was invoked
        - `graph["positions"]`: hierarchical layout positions keyed by graph
          node id

        Graph node ids are `id(node)` for the corresponding `Node` object.
        Each graph node stores:

        - `node`: original `Node` instance
        - `label`: node name
        - `target_key`: node `_target`
        - `is_current`: whether this is the requested node
        - `is_root`: whether this is the root node

        Each directed edge stores:

        - `target`: leaf lookup key on the parent
        - `transform`: corresponding `_LeavePath.transform`

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
            target_key=current._target,
            is_current=current is node,
            is_root=current is root,
        )
        for target, path in current._leaves.items():
            child = path.node
            child_id = id(child)
            graph.add_node(
                child_id,
                node=child,
                label=child.name,
                target_key=child._target,
                is_current=child is node,
                is_root=False,
            )
            graph.add_edge(current_id, child_id, target=target, transform=path.transform)
            stack.append(child)

    return graph


@Node.register_plot_method("tree", backend="matplotlib")  # type: ignore[untyped-decorator]
def _(
    node: Node,
    *,
    ax: Any = None,
    figsize: tuple[float, float] = (10.0, 6.0),
    label_width: int = 18,
) -> Any:
    """
    Render the full node tree as a matplotlib figure with wrapped node labels.

    Parameters
    ----------
    node : Node
        Current node from which plotting was requested.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If omitted, a new figure and axes are
        created.
    figsize : tuple[float, float], default=(10.0, 6.0)
        Figure size used when `ax` is not provided.
    label_width : int, default=18
        Maximum character width before node labels are wrapped onto new lines.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the rendered tree.

    Notes
    -----
    The current node is highlighted in crimson, the root in gold, and all
    other nodes in light blue. Labels are wrapped before being placed inside
    rounded bounding boxes so long names fit within node shapes more cleanly.
    Connectors are drawn as orthogonal rounded-corner arrows so branching
    structure reads more like a diagram than a raw graph.
    """
    graph = _plot_node_tree_networkx(node)
    positions = graph.graph["positions"]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
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

        connector_path = MplPath(
            [
                (x0, y_start),
                (x0, junction_y),
                (x1, junction_y),
                (x1, y_end),
            ],
            [
                MplPath.MOVETO,
                MplPath.LINETO,
                MplPath.LINETO,
                MplPath.LINETO,
            ],
        )
        connector = PathPatch(
            connector_path,
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
        )

        label = _wrap_node_label(attrs["label"], attrs["target_key"], width=label_width)
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
def _(
    node: Node,
    *,
    figure: Any = None,
    label_width: int = 18,
) -> Any:
    """
    Render the full node tree as an interactive Plotly figure.

    Parameters
    ----------
    node : Node
        Current node from which plotting was requested.
    figure : plotly.graph_objects.Figure, optional
        Existing figure to populate. If omitted, a new figure is created.
    label_width : int, default=18
        Maximum character width before node labels are wrapped onto new lines.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing the rendered tree.
    """
    graph = _plot_node_tree_networkx(node)
    positions = graph.graph["positions"]

    if figure is None:
        fig = go.Figure()
    else:
        fig = figure

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
        )
        label = _wrap_node_label(
            cast(str, attrs["label"]),
            cast(str, attrs["target_key"]),
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
            font={
                "color": style["textcolor"],
                "size": 11,
                "family": "Courier New, monospace",
            },
            bgcolor=style["facecolor"],
            bordercolor=style["edgecolor"],
            borderwidth=style["linewidth"],
            borderpad=10,
        )
        node_x.append(x)
        node_y.append(y)
        hover_text.append(
            f"name: {cast(str, attrs['label'])}<br>target: {cast(str, attrs['target_key'])}"
        )
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
                "line": {
                    "color": marker_line_color,
                    "width": marker_line_width,
                },
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
        xaxis={
            "range": [min(xs) - 0.95, max(xs) + 0.95],
            "visible": False,
        },
        yaxis={
            "range": [min(ys) - 0.9, max(ys) + 0.9],
            "visible": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig
