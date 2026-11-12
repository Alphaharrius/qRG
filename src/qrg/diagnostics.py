"""Named read-only checks that record a result on a node."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_DIAGNOSTICS: dict[str, Callable[..., Any]] = {}


def diagnostic(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register `fn` under `name`.

    The function receives a node and keyword arguments, and returns the value
    to store.

    Raises
    ------
    ValueError
        If `name` is empty or already registered.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name == "":
            raise ValueError("Diagnostic name must be a non-empty string")
        if name in _DIAGNOSTICS:
            raise ValueError(f"Diagnostic {name!r} is already registered")
        _DIAGNOSTICS[name] = fn
        return fn

    return decorator


def unregister(name: str) -> None:
    """Remove a diagnostic if it is present."""
    _DIAGNOSTICS.pop(name, None)


def names() -> tuple[str, ...]:
    """Return registered diagnostic names in insertion order."""
    return tuple(_DIAGNOSTICS)


def run(node: Any, name: str, /, **params: Any) -> Any:
    """
    Evaluate a diagnostic and store the result on `node`.

    The value is written with :meth:`qrg.tree.Node.record` under `name`.

    Raises
    ------
    ValueError
        If `name` is not registered.
    """
    try:
        fn = _DIAGNOSTICS[name]
    except KeyError:
        raise ValueError(f"Diagnostic {name!r} is not registered") from None
    result = fn(node, **params)
    node.record(name, result)
    return result
