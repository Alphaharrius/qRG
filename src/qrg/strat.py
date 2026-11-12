"""
Renormalization strategies.

A strategy is a small data object: an implementation name plus parameters.
:meth:`qrg.tree.Node.add_strat` stores it on the tree. :meth:`qrg.tree.Node.grow`
looks the implementation up here and appends one layer.

::

    root.add_strat("blocking", normal_blocking(2))
    root.add_strat("wannier", proj_wannierization(frozen="occupied", courier="rest"))
    root.add_strat("upscale", upscale(2))
    layer = root
    for _ in range(3):
        layer = layer.grow("blocking").grow("wannier").grow("upscale")
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

from .rg import project
from .tree import OPERATOR, Branch, Node, Strategy, TensorLike

F = TypeVar("F", bound=Callable[..., Branch])

_IMPLS: dict[str, Callable[..., Branch]] = {}


def register(name: str, fn: Callable[..., Branch]) -> None:
    """Register or replace a strategy implementation."""
    if name == "":
        raise ValueError("Strategy implementation name must be a non-empty string")
    _IMPLS[name] = fn


def unregister(name: str) -> None:
    """Remove a strategy implementation if it is present."""
    _IMPLS.pop(name, None)


def get_impl(name: str) -> Callable[..., Branch]:
    """
    Return the implementation registered as `name`.

    Raises
    ------
    ValueError
        If `name` is not registered.
    """
    try:
        return _IMPLS[name]
    except KeyError:
        raise ValueError(f"Strategy implementation {name!r} is not registered") from None


def _implementation(name: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        register(name, fn)
        return fn

    return decorator


def _check_factor(factor: int | Sequence[int]) -> int | tuple[int, ...]:
    if isinstance(factor, bool):
        raise ValueError(f"Blocking factor must be a positive integer, got {factor!r}")
    if isinstance(factor, int):
        if factor < 1:
            raise ValueError(f"Blocking factor must be a positive integer, got {factor!r}")
        return factor
    if isinstance(factor, str):
        raise ValueError(f"Blocking factor must be an int or a sequence of ints, got {factor!r}")
    values = tuple(int(value) for value in factor)
    if not values or any(value < 1 for value in values):
        raise ValueError(f"Blocking factor must be positive along every axis, got {factor!r}")
    return values


def _lattice_dim(operator: Any) -> int:
    from qten.geometries import ReciprocalLattice

    lattice = operator.dims[0].extract(ReciprocalLattice)
    return int(lattice.dim)


def _axis_scales(operator: Any, factor: int | tuple[int, ...], *, step: str) -> tuple[int, ...]:
    dim = _lattice_dim(operator)
    if isinstance(factor, int):
        return (factor,) * dim
    scales = tuple(factor)
    if len(scales) != dim:
        raise ValueError(f"{step} factor has length {len(scales)} for a {dim}-dimensional lattice")
    return scales


def normal_blocking(factor: int | Sequence[int]) -> Strategy:
    """
    Isotropic or per-axis band folding.

    `normal_blocking(3)` folds every lattice axis by 3. `normal_blocking((2, 2, 1))`
    sets one factor per axis. The folded Hamiltonian becomes the child
    operator. This step has no isometry.
    """
    stored = _check_factor(factor)
    return Strategy("normal_blocking", {"factor": stored})


def proj_wannierization(*, frozen: str, courier: str, **params: Any) -> Strategy:
    """
    Project the current operator with two named seed recipes.

    Extra keyword arguments are stored on the edge and forwarded to both
    recipes. The child operator is the courier Hamiltonian.
    ``node["frozen"]`` and ``node["courier"]`` hold the two isometries.
    """
    if frozen == "" or courier == "":
        raise ValueError("frozen and courier seed names must be non-empty")
    stored: dict[str, Any] = {"frozen": frozen, "courier": courier, **params}
    return Strategy("proj_wannierization", stored)


@_implementation("normal_blocking")
def _normal_blocking(node: Node, *, factor: int | tuple[int, ...]) -> Branch:
    from qten.bands import bandfold
    from qten.geometries import BasisTransform
    from sympy import ImmutableDenseMatrix, Integer

    scales = _axis_scales(node.operator(), factor, step="Blocking")
    dim = len(scales)
    matrix = ImmutableDenseMatrix(
        [
            [Integer(scales[row]) if row == col else Integer(0) for col in range(dim)]
            for row in range(dim)
        ]
    )
    folded = bandfold(BasisTransform(matrix), node.operator(), opt="both")
    return Branch(
        data={OPERATOR: cast(TensorLike, folded)},
        info={"factor": scales if not isinstance(factor, int) else factor},
        kind="blocking",
    )


@_implementation("proj_wannierization")
def _proj_wannierization(node: Node, *, frozen: str, courier: str, **params: Any) -> Branch:
    from .seeds import materialize

    operator = node.operator()
    frozen_w, frozen_info = materialize(frozen, operator, **params)
    courier_w, courier_info = materialize(courier, operator, frozen_isometry=frozen_w, **params)
    return Branch(
        data={
            OPERATOR: cast(TensorLike, project(operator, courier_w)),
            "frozen": cast(TensorLike, frozen_w),
            "courier": cast(TensorLike, courier_w),
        },
        transform=cast(TensorLike, courier_w),
        info={"frozen": frozen_info, "courier": courier_info},
        kind="isometry",
    )


def upscale(factor: int | Sequence[int]) -> Strategy:
    """
    Interpolate the current operator onto a denser momentum grid.

    `upscale(2)` multiplies every real-space period by 2, so the Brillouin-zone
    mesh grows by that factor along each axis. Orbital content is unchanged.
    Use it after a courier projection, when folding has coarsened the mesh and
    the next blocking step needs the fine grid back. `upscale((2, 1))` sets one
    factor per axis.

    This calls :func:`qten.bands.upsample`. It does not undo a fold and it does
    not restore symmetry; a model-specific average stays outside this step.
    """
    stored = _check_factor(factor)
    return Strategy("upscale", {"factor": stored})


@_implementation("upscale")
def _upscale(node: Node, *, factor: int | tuple[int, ...]) -> Branch:
    from qten.bands import upsample

    scales = _axis_scales(node.operator(), factor, step="Upscale")
    enlarged = upsample(node.operator(), scales)
    return Branch(
        data={OPERATOR: cast(TensorLike, enlarged)},
        info={"factor": factor if isinstance(factor, int) else scales},
        kind="upscale",
    )
