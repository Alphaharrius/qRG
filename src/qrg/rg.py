"""
Projection helpers used by growth methods.

The tree core does not import this module. A growth method owns the physics:
it builds isometries, then calls :func:`subspaces` to package them as
branches. Folding a Hamiltonian is not a matrix product, so those branches
should use a ``kind`` other than ``"isometry"``.

The built-in steps are :func:`qrg.strat.normal_blocking` and
:func:`qrg.strat.proj_wannierization`. ``project`` is the ``W† H W`` piece
those steps use on the operator of the node they are called on.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .tree import OPERATOR, Branch, TensorLike


def project(operator: Any, transform: Any) -> Any:
    """
    Project an operator through an isometry.

    Returns ``transform.h(-2, -1) @ operator @ transform``, the operator on
    the child space.
    """
    return transform.h(-2, -1) @ operator @ transform


def lift(operator: Any, transform: Any) -> Any:
    """
    Push a child-space operator back onto the parent space.

    Returns ``transform @ operator @ transform.h(-2, -1)``.
    """
    return transform @ operator @ transform.h(-2, -1)


def subspaces(
    operator: Any,
    parts: Mapping[str, Any],
    *,
    extra: Mapping[str, Mapping[str, Any]] | None = None,
    info: Mapping[str, Mapping[str, Any]] | None = None,
    kind: str = "isometry",
) -> dict[str, Branch]:
    """
    Build one branch per isometry, each carrying the projected operator.

    Parameters
    ----------
    operator : qten.Tensor
        Operator on the space the isometries start from. For a folded step
        this is the folded Hamiltonian, not the parent node's operator.
    parts : mapping of str to qten.Tensor
        Branch name to isometry. Each isometry's column space is the child.
    extra : mapping, optional
        Extra tensors for each branch, keyed by branch name. ``"H"`` is
        reserved for the projected operator.
    info : mapping, optional
        Non-tensor records for each branch, keyed by branch name.
    kind : str, default="isometry"
        Stored on every branch. Use another name when `operator` is not the
        parent node's ``"H"``.

    Returns
    -------
    dict of str to Branch
        Branches ready to return from a growth method.

    Raises
    ------
    ValueError
        If `extra` or `info` names a branch that is not in `parts`, or if an
        extra tensor uses the key ``"H"``.
    """
    extras: Mapping[str, Mapping[str, Any]] = {} if extra is None else extra
    notes: Mapping[str, Mapping[str, Any]] = {} if info is None else info
    unknown = sorted((set(extras) | set(notes)) - set(parts))
    if unknown:
        raise ValueError(f"Keys {unknown} do not match a branch: {sorted(parts)}")

    branches: dict[str, Branch] = {}
    for name, transform in parts.items():
        payload: dict[str, TensorLike] = {OPERATOR: cast(TensorLike, project(operator, transform))}
        for key, value in extras.get(name, {}).items():
            if key == OPERATOR:
                raise ValueError("extra tensors must not include 'H'; it is the projected operator")
            payload[key] = cast(TensorLike, value)
        recorded = notes.get(name)
        if recorded is None:
            recorded = {}
        branches[name] = Branch(
            transform=cast(TensorLike, transform),
            data=payload,
            info=dict(recorded),
            kind=kind,
        )
    return branches
