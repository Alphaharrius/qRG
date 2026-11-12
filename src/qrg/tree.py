"""
Tree of renormalization steps.

Each :meth:`Node.grow` call appends one layer. The layer's strategy is chosen
earlier with :meth:`Node.add_strat` and stored on the root as data, not as a
function, so the tree can be pickled. Strategy implementations live in
:mod:`qrg.strat`.

The operator on every node lives under ``"H"``. Extra tensors from that step,
such as a frozen Wannier function, live in the same mapping.
"""

from __future__ import annotations

import re
import traceback
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import qten

TensorLike = qten.Tensor[Any]
OPERATOR = "H"


@dataclass(frozen=True, eq=False)
class Strategy:
    """
    A named implementation plus the parameters to call it with.

    Parameters
    ----------
    impl : str
        Implementation name registered in :mod:`qrg.strat`.
    params : mapping
        Keyword arguments forwarded to that implementation. Use strings,
        numbers, and tuples so the strategy survives pickling.
    """

    impl: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.impl == "":
            raise ValueError("Strategy.impl must be a non-empty string")
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True, eq=False)
class Branch:
    """
    The single layer a strategy produces, before it is attached to the tree.

    Parameters
    ----------
    data : mapping of str to qten.Tensor
        Tensors owned by the new node. Must contain ``"H"``.
    transform : qten.Tensor or None, optional
        Isometry stored on the edge. Folding steps leave this empty.
    info : mapping of str to object, optional
        Non-tensor records such as overlap or a blocking factor.
    kind : str, default="isometry"
        ``"isometry"`` means `transform` maps the parent operator onto ``"H"``.
        Any other kind is skipped by
        [`get_transform`][qrg.tree.Node.get_transform].
    name : str or None, optional
        Display name. Defaults to the strategy name passed to
        [`grow`][qrg.tree.Node.grow].
    """

    data: Mapping[str, TensorLike]
    transform: TensorLike | None = None
    info: Mapping[str, Any] = field(default_factory=dict)
    kind: str = "isometry"
    name: str | None = None

    def __post_init__(self) -> None:
        if OPERATOR not in self.data:
            raise ValueError("Branch.data must contain the operator under 'H'")
        if self.kind == "":
            raise ValueError("Branch.kind must be a non-empty string")
        object.__setattr__(self, "data", dict(self.data))
        object.__setattr__(self, "info", dict(self.info))


@dataclass(frozen=True, eq=False)
class Step:
    """
    Provenance of the edge that created a node.

    Parameters
    ----------
    method : str
        Strategy name used with [`grow`][qrg.tree.Node.grow], such as
        ``"blocking"``. This is also the child-key in paths.
    impl : str
        Implementation name, such as ``"normal_blocking"``.
    params : mapping
        Parameters stored on the strategy.
    transform : qten.Tensor or None
        Isometry copied from the branch. ``None`` when the step is not an
        isometry.
    kind : str
        Copied from the branch. See [`Branch`][qrg.tree.Branch].
    """

    method: str
    impl: str
    params: Mapping[str, Any]
    transform: TensorLike | None = None
    kind: str = "isometry"

    def __post_init__(self) -> None:
        if self.method == "":
            raise ValueError("Step.method must be a non-empty string")
        if self.impl == "":
            raise ValueError("Step.impl must be a non-empty string")
        if self.kind == "":
            raise ValueError("Step.kind must be a non-empty string")
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True, eq=False)
class Leaf:
    """
    Direct child of a node, together with the transform on that edge.

    Parameters
    ----------
    node : Node
        Child node.
    transform : qten.Tensor or None
        Isometry stored on the child's step.
    """

    node: Node
    transform: TensorLike | None


class Node(Mapping[str, TensorLike], qten.plottings.Plottable):
    """
    One space in a renormalization tree.

    The mapping interface reads and writes the tensors owned by this node.
    ``node["H"]`` is the operator on this space, also available as
    [`operator()`][qrg.tree.Node.operator]. Child spaces are created only by
    [`grow()`][qrg.tree.Node.grow].

    Parameters
    ----------
    name : str
        Display name.
    _data : dict
        Tensors owned by the node, including ``"H"``.
    _info : dict
        Non-tensor records.
    _parent : Node or None
        Parent node. ``None`` at the root.
    _key : str or None
        Lookup key of this node under its parent.
    _step : Step or None
        Edge that created this node. ``None`` at the root.
    _children : dict
        Direct children, keyed by the strategy name passed to
        [`grow`][qrg.tree.Node.grow].
    _strats : dict
        Strategies configured for this tree. Only the root's dictionary is
        used. [`add_strat`][qrg.tree.Node.add_strat] writes there from any node.
    _history : list
        Run log. Only the root's list is appended to. Each [`grow`][qrg.tree.Node.grow]
        and each exported figure adds one entry.
    """

    name: str
    _data: dict[str, TensorLike]
    _info: dict[str, Any]
    _parent: Node | None
    _key: str | None
    _step: Step | None
    _children: dict[str, Node]
    _strats: dict[str, Strategy]
    _history: list[dict[str, Any]]

    def __init__(
        self,
        name: str,
        _data: dict[str, TensorLike],
        _info: dict[str, Any],
        _parent: Node | None,
        _key: str | None,
        _step: Step | None,
        _children: dict[str, Node],
        _strats: dict[str, Strategy],
        _history: list[dict[str, Any]],
    ) -> None:
        self.name = name
        self._data = _data
        self._info = _info
        self._parent = _parent
        self._key = _key
        self._step = _step
        self._children = _children
        self._strats = _strats
        self._history = _history

    def __repr__(self) -> str:
        step = None if self._step is None else self._step.method
        return (
            f"Node(name={self.name!r}, step={step!r}, "
            f"data={list(self._data)}, children={list(self._children)})"
        )

    def __getitem__(self, key: str) -> TensorLike:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __setitem__(self, key: str, value: TensorLike) -> None:
        self._data[key] = value

    @property
    def step(self) -> Step | None:
        """Edge that created this node, or ``None`` at the root."""
        return self._step

    @property
    def children(self) -> Mapping[str, Node]:
        """Direct children, in the order they were attached."""
        return MappingProxyType(self._children)

    @property
    def info(self) -> Mapping[str, Any]:
        """Non-tensor records stored on this node."""
        return MappingProxyType(self._info)

    @property
    def strats(self) -> Mapping[str, Strategy]:
        """Strategies configured on the root of this tree."""
        return MappingProxyType(self.root()._strats)

    def add_strat(self, name: str, strategy: Strategy) -> Self:
        """
        Configure a strategy on the root of this tree.

        Parameters
        ----------
        name : str
            Name used later with [`grow`][qrg.tree.Node.grow]. It becomes the
            child key, so it cannot be empty or contain a dot.
        strategy : Strategy
            Implementation and parameters. Factories in :mod:`qrg.strat` return
            these objects.

        Returns
        -------
        Node
            This node, so calls can be chained.
        """
        if name == "" or "." in name:
            raise ValueError(f"Strategy name {name!r} cannot be empty or contain '.'")
        if not isinstance(strategy, Strategy):
            raise TypeError(
                "add_strat expects a Strategy, for example normal_blocking(2) "
                "or proj_wannierization(frozen=..., courier=...)"
            )
        self.root()._strats[name] = strategy
        return self

    @property
    def history(self) -> tuple[Mapping[str, Any], ...]:
        """
        Records of every grow and figure export on this tree.

        Each entry has ``kind`` (``"grow"`` or ``"export"``), ``time``, and
        ``depth``. Grow entries also keep the strategy name, implementation,
        parameters, path, and the layer's non-tensor ``info``.
        """
        return tuple(MappingProxyType(dict(entry)) for entry in self.root()._history)

    def plot(
        self,
        method: str,
        backend: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> object:
        """
        Dispatch a plot, defaulting ``tree`` plots to matplotlib.

        Parameters
        ----------
        method : str
            Plot method name.
        backend : str or None, optional
            Backend name. Omitted ``tree`` plots use matplotlib. Other plots
            keep the plotly default.
        export : path, optional
            Write the figure to this file and append an ``export`` record to
            [`history`][qrg.tree.Node.history]. ``.png``, ``.pdf``, and
            ``.svg`` use matplotlib's ``savefig`` when the figure has one.
            ``.html`` uses Plotly's ``write_html``.
        """
        export = kwargs.pop("export", None)
        if backend is None or backend == "":
            backend = "matplotlib" if method == "tree" else "plotly"
        figure = super().plot(method, backend, *args, **kwargs)
        if export is not None:
            saved = _export_figure(figure, export)
            self._remember(
                {
                    "kind": "export",
                    "plot": method,
                    "backend": backend,
                    "file": str(saved),
                    "path": self.path(),
                    "depth": self.depth(),
                }
            )
        return figure

    def operator(self) -> TensorLike:
        """Return the operator stored under ``"H"``."""
        if OPERATOR not in self._data:
            err = self._info.get("error")
            if isinstance(err, Mapping):
                raise RuntimeError(
                    f"Node {self.path() or self.name!r} has no operator because it failed "
                    f"with {err.get('type', 'Error')}: {err.get('message', '')}"
                )
            raise KeyError(OPERATOR)
        return self._data[OPERATOR]

    @property
    def failed(self) -> bool:
        """Return ``True`` if this node failed or recorded an error."""
        if "error" in self._info:
            return True
        return self._step is not None and self._step.kind == "error"

    def record_error(self, exc: BaseException, *, stage: str | None = None) -> Self:
        """Record `exc` on this node so reports display an ``ERROR`` block."""
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        payload: dict[str, Any] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb,
        }
        if stage:
            payload["stage"] = stage
        self._info["error"] = payload
        self._remember(
            {
                "kind": "error",
                "path": self.path(),
                "depth": self.depth(),
                "stage": stage or "record_error",
                "error": dict(payload),
            }
        )
        return self

    def fail(
        self,
        name: str,
        exc: BaseException,
        *,
        impl: str = "external",
        params: Mapping[str, Any] | None = None,
    ) -> Node:
        """Attach and return an ``ERROR`` child node representing a failed step."""
        if name == "" or "." in name:
            raise ValueError(f"Branch name {name!r} cannot be empty or contain '.'")
        key = name
        suffix = 2
        while key in self._children:
            key = f"{name}_{suffix}"
            suffix += 1
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        err_payload: dict[str, Any] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb,
        }
        stored_params = {} if params is None else dict(params)
        child = Node(
            name=name,
            _data={},
            _info={"error": err_payload},
            _parent=self,
            _key=key,
            _step=Step(
                method=key,
                impl=impl,
                params=stored_params,
                transform=None,
                kind="error",
            ),
            _children={},
            _strats={},
            _history=[],
        )
        self._children[key] = child
        self._remember(
            {
                "kind": "error",
                "path": child.path(),
                "depth": child.depth(),
                "strategy": key,
                "impl": impl,
                "params": stored_params,
                "error": dict(err_payload),
            }
        )
        return child

    def compute(self, name: str, f: Callable[[Node], TensorLike]) -> Self:
        """
        Store ``f(self)`` in the tensor mapping and return this node.

        Parameters
        ----------
        name : str
            Key to write.
        f : callable
            Receives this node and returns a tensor.
        """
        self._data[name] = f(self)
        return self

    def record(self, key: str, value: Any) -> Self:
        """
        Store a non-tensor value in [`info`][qrg.tree.Node.info].

        Parameters
        ----------
        key : str
            Record name.
        value : object
            JSON-like or otherwise picklable value. Large tensors belong in
            the tensor mapping, not here.
        """
        self._info[key] = value
        return self

    def leaf(self, name: str) -> Leaf:
        """
        Return a direct child and the transform on its edge.

        Raises
        ------
        ValueError
            If `name` is not a direct child.
        """
        try:
            child = self._children[name]
        except KeyError:
            raise ValueError(f"Target {name} not found in leaves of node {self.name}") from None
        if child._step is None:
            raise RuntimeError(f"Child {name!r} of node {self.name!r} has no step")
        return Leaf(node=child, transform=child._step.transform)

    def branches(self) -> dict[str, Leaf]:
        """Return a copy of the direct children and their edge transforms."""
        return {name: self.leaf(name) for name in self._children}

    def cut(self, name: str) -> Node:
        """
        Detach a direct child and return it as its own root.

        The detached node drops its parent link and the edge that created it.
        Tensors and its own children stay.

        Raises
        ------
        ValueError
            If `name` is not a direct child.
        """
        try:
            child = self._children.pop(name)
        except KeyError:
            raise ValueError(f"Target {name} not found in leaves of node {self.name}") from None
        child._strats = dict(self.root()._strats)
        child._history = [dict(entry) for entry in self.root()._history]
        child._parent = None
        child._key = None
        child._step = None
        return child

    def trace(self, expr: str | int, *, relative: bool = False) -> Node:
        """
        Resolve a node by path or by how many layers to move.

        String paths:

        - ``"a.b"`` is absolute and starts at the root.
        - ``".a.b"`` is relative and starts at this node.
        - ``""`` is the root and ``"."`` is this node.
        - ``relative=True`` treats a string without a leading dot as relative.

        Counts:

        - ``trace(0)`` is the root. ``trace(2)`` is the node two layers down
          from the root. Each step must have exactly one child.
        - ``trace(0, relative=True)`` is this node. A positive relative count
          walks that many unique children. A negative count walks parents.
        - ``trace(-1)`` is the parent. Negative counts are always relative.

        [`depth`][qrg.tree.Node.depth] is the absolute layer number of this node.

        Raises
        ------
        ValueError
            If a path segment is missing, or a count tries to pass a layer
            that does not have exactly one child.
        TypeError
            If `expr` is not a string or an integer.
        """
        if isinstance(expr, bool):
            raise TypeError("trace() expects a path or an integer, got bool")
        if isinstance(expr, int):
            if relative or expr < 0:
                return self._trace_count(expr)
            return self.root()._trace_count(expr)
        if not isinstance(expr, str):
            raise TypeError(f"trace() expects a path or an integer, got {type(expr).__name__}")
        if relative and not expr.startswith("."):
            expr = "." if expr == "" else f".{expr}"
        return self._trace_path(expr)

    def _trace_path(self, expr: str) -> Node:
        is_relative = expr.startswith(".")
        raw_path = expr[1:] if is_relative else expr
        current = self if is_relative else self.root()
        if raw_path == "":
            return current

        parts = raw_path.split(".")
        if not parts or any(part == "" for part in parts):
            raise ValueError(f'Invalid trace expression "{expr}"')
        for index, part in enumerate(parts):
            if part not in current._children:
                resolved = parts[:index]
                unresolved = parts[index + 1 :]
                highlighted = ".".join([*resolved, f"[{part}]", *unresolved])
                raise ValueError(f'trace cannot be resolve at "{highlighted}"')
            current = current._children[part]
        return current

    def _trace_count(self, steps: int) -> Node:
        if steps < 0:
            current = self
            for _ in range(-steps):
                current = current.parent()
            return current
        current = self
        for _ in range(steps):
            children = list(current._children)
            if len(children) != 1:
                located = current.path() or "."
                raise ValueError(
                    f"Node {located!r} has {len(children)} children; "
                    "use a path string to choose one"
                )
            current = current._children[children[0]]
        return current

    def path(self) -> str:
        """
        Return the child-key path from the root to this node.

        The root path is ``""``.

        Raises
        ------
        RuntimeError
            If a parent link does not point back at this node.
        """
        parts: list[str] = []
        current = self
        while current._parent is not None:
            parent = current._parent
            key = current._key
            if key is None or parent._children.get(key) is not current:
                raise RuntimeError(
                    f"Node {current.name!r} is not reachable from parent {parent.name!r}"
                )
            parts.append(key)
            current = parent
        parts.reverse()
        return ".".join(parts)

    def depth(self) -> int:
        """Return how many layers this node is below the root. The root is 0."""
        count = 0
        current = self
        while current._parent is not None:
            count += 1
            current = current._parent
        return count

    def get_transform(
        self, expr: str | int, *, composed: bool = True, relative: bool = False
    ) -> TensorLike | list[TensorLike]:
        """
        Return the isometry from this node to a descendant.

        Parameters
        ----------
        expr : str or int
            Path or layer count understood by [`trace`][qrg.tree.Node.trace].
        relative : bool, default=False
            Forwarded to [`trace`][qrg.tree.Node.trace].
        composed : bool, default=True
            If ``True``, return ``T1 @ T2 @ ... @ Tn``. If ``False``, return
            that list of edge transforms.

        Returns
        -------
        qten.Tensor or list of qten.Tensor
            The identity on this node's operator when the path is empty.

        Raises
        ------
        ValueError
            If the path leaves this subtree, or any edge has
            ``kind != "isometry"``. Folded renormalization steps need their
            own transport helper. They are not one matrix product.

        Notes
        -----
        For an isometry chain, the descendant operator is
        ``T.h(-2, -1) @ operator @ T``.
        """
        destination = self.trace(expr, relative=relative)
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
            edge = current.leaf(part)
            step = edge.node.step
            if step is None or step.kind != "isometry" or step.transform is None:
                kind = None if step is None else step.kind
                raise ValueError(
                    f"Cannot compose transforms at {edge.node.path()!r}: "
                    f"step kind is {kind!r}, not 'isometry'"
                )
            transforms.append(step.transform)
            current = edge.node

        if not composed:
            return transforms
        if not transforms:
            current_operator = self.operator()
            return qten.eye(current_operator.dims, device=current_operator.device)
        transform = transforms[0]
        for piece in transforms[1:]:
            transform = transform @ piece
        return transform

    def find(
        self,
        *,
        regex: str | None = None,
        predicate: Callable[[Node], bool] | None = None,
    ) -> list[Node]:
        """
        Return nodes in this subtree whose name matches `regex` and `predicate`.

        The walk is depth-first and includes this node. When both filters are
        given, a node must pass both.

        Raises
        ------
        ValueError
            If neither filter is given.
        """
        if regex is None and predicate is None:
            raise ValueError("find() requires regex and/or predicate")

        pattern = re.compile(regex) if regex is not None else None
        matches: list[Node] = []
        stack = [self]
        while stack:
            current = stack.pop()
            regex_match = pattern.search(current.name) is not None if pattern is not None else True
            predicate_match = predicate(current) if predicate is not None else True
            if regex_match and predicate_match:
                matches.append(current)
            stack.extend(reversed(list(current._children.values())))
        return matches

    def parent(self) -> Node:
        """
        Return the parent node.

        Raises
        ------
        ValueError
            If this node is the root.
        """
        if self._parent is None:
            raise ValueError(f"Node {self.name} has no parent")
        return self._parent

    def root(self) -> Node:
        """Return the top ancestor, or this node when it is the root."""
        root = self
        while root._parent is not None:
            root = root._parent
        return root

    def is_root(self) -> bool:
        """Return whether this node has no parent."""
        return self._parent is None

    def is_leaf(self) -> bool:
        """Return whether this node has no children."""
        return not self._children

    def report(self, path: str | Path, *, selected: Node | str | None = None) -> Path:
        """
        Write an HTML page for this tree and return the file path.

        The page opens on the latest leaf by default, or on `selected` when
        given. The link into each node is the step that created it, and the
        node shows the tensors, tables, and plots recorded there.
        """
        from .report import write_report

        return write_report(self, path, selected=selected)

    def grow(self, name: str, *, on_error: str = "raise") -> Node:
        """
        Append the layer configured under `name` and return that new node.

        The strategy is read from the root, including when this node is deeper
        in the tree. A second ``grow("blocking")`` on the returned node starts
        the next renormalization step.

        Parameters
        ----------
        name : str
            Name passed to [`add_strat`][qrg.tree.Node.add_strat].
        on_error : {"raise", "record"}, default="raise"
            When ``"record"``, any runtime exception raised by the strategy
            implementation is captured into an ``ERROR`` child node instead of
            aborting execution, so partial trees can still render a report.

        Returns
        -------
        Node
            The new child. Its path segment is `name`.

        Raises
        ------
        ValueError
            If `name` was not configured, its implementation is missing,
            `on_error` is invalid, or this node already has a child with that key.
        TypeError
            If the implementation does not return a :class:`Branch`.
        """
        from .strat import get_impl

        if on_error not in {"raise", "record"}:
            raise ValueError(f"on_error must be 'raise' or 'record', got {on_error!r}")
        try:
            strategy = self.root()._strats[name]
        except KeyError:
            raise ValueError(f"Strategy {name!r} is not set on this tree") from None
        if name in self._children:
            raise ValueError(f"Node {self.name!r} already has a branch {name!r}")

        impl_fn = get_impl(strategy.impl)
        try:
            if self.failed:
                raise RuntimeError(
                    f"Cannot grow {name!r} from failed parent node {self.path() or self.name!r}"
                )
            produced = impl_fn(self, **dict(strategy.params))
            if not isinstance(produced, Branch):
                raise TypeError(
                    f"Strategy {strategy.impl!r} must return Branch, got {type(produced).__name__}"
                )
        except Exception as exc:
            if on_error == "raise":
                raise
            return self.fail(name, exc, impl=strategy.impl, params=strategy.params)

        child = Node(
            name=name if produced.name is None else produced.name,
            _data=dict(produced.data),
            _info=dict(produced.info),
            _parent=self,
            _key=name,
            _step=Step(
                method=name,
                impl=strategy.impl,
                params=strategy.params,
                transform=produced.transform,
                kind=produced.kind,
            ),
            _children={},
            _strats={},
            _history=[],
        )
        self._children[name] = child
        self._remember(
            {
                "kind": "grow",
                "path": child.path(),
                "depth": child.depth(),
                "strategy": name,
                "impl": strategy.impl,
                "params": dict(strategy.params),
                "info": dict(child._info),
            }
        )
        return child

    def _remember(self, entry: dict[str, Any]) -> None:
        entry["time"] = datetime.now(timezone.utc).isoformat()  # noqa: UP017
        self.root()._history.append(entry)

    @staticmethod
    def new(
        *,
        name: str,
        operator: TensorLike,
        data: Mapping[str, TensorLike] | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> Node:
        """
        Create a root node whose operator is stored under ``"H"``.

        Parameters
        ----------
        name : str
            Root name.
        operator : qten.Tensor
            Hamiltonian, or whichever operator later steps project.
        data : mapping of str to qten.Tensor, optional
            Extra tensors. Must not include ``"H"``.
        info : mapping, optional
            Non-tensor records.
        """
        payload: dict[str, TensorLike] = {OPERATOR: operator}
        if data is not None:
            if OPERATOR in data:
                raise ValueError("data must not include 'H'; pass it as operator")
            payload.update(data)
        return Node(
            name=name,
            _data=payload,
            _info={} if info is None else dict(info),
            _parent=None,
            _key=None,
            _step=None,
            _children={},
            _strats={},
            _history=[],
        )

    @staticmethod
    def failed_root(
        *,
        name: str,
        error: BaseException,
        info: Mapping[str, Any] | None = None,
    ) -> Node:
        """Create a root ``ERROR`` node when initial operator construction fails."""
        node = Node(
            name=name,
            _data={},
            _info={} if info is None else dict(info),
            _parent=None,
            _key=None,
            _step=None,
            _children={},
            _strats={},
            _history=[],
        )
        node.record_error(error, stage="init")
        return node


def _export_figure(figure: object, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    savefig = getattr(figure, "savefig", None)
    if callable(savefig) and suffix in {".png", ".pdf", ".svg"}:
        savefig(destination, bbox_inches="tight")
        return destination
    write_html = getattr(figure, "write_html", None)
    if callable(write_html) and suffix in {".html", ".htm"}:
        write_html(destination)
        return destination
    write_image = getattr(figure, "write_image", None)
    if callable(write_image):
        write_image(destination)
        return destination
    raise TypeError(f"Cannot export {type(figure).__name__} to {destination}")


from . import _plot_tree as _plot_tree  # noqa: E402

del _plot_tree
