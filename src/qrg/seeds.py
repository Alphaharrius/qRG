"""Named recipes that build trial states for a strategy step."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

_SEEDS: dict[str, Callable[..., Any]] = {}


def seed(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Register `fn` under `name`.

    Raises
    ------
    ValueError
        If `name` is empty or already registered.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name == "":
            raise ValueError("Seed name must be a non-empty string")
        if name in _SEEDS:
            raise ValueError(f"Seed {name!r} is already registered")
        _SEEDS[name] = fn
        return fn

    return decorator


def unregister(name: str) -> None:
    """Remove a seed recipe if it is present."""
    _SEEDS.pop(name, None)


def names() -> tuple[str, ...]:
    """Return registered seed names in insertion order."""
    return tuple(_SEEDS)


def make(name: str, /, *args: Any, **kwargs: Any) -> Any:
    """
    Call the recipe registered as `name`.

    Raises
    ------
    ValueError
        If `name` is not registered.
    """
    try:
        fn = _SEEDS[name]
    except KeyError:
        raise ValueError(f"Seed {name!r} is not registered") from None
    return fn(*args, **kwargs)


def materialize(name: str, operator: Any, **params: Any) -> tuple[Any, dict[str, Any]]:
    """
    Build the trial tensor for `operator`.

    A recipe may return a tensor, or ``(tensor, info)`` where `info` is a
    mapping of non-tensor records. The second form is what
    :func:`qrg.strat.proj_wannierization` stores on the new node.
    """
    produced = make(name, operator, **params)
    if isinstance(produced, tuple) and len(produced) == 2 and isinstance(produced[1], Mapping):
        return produced[0], dict(produced[1])
    return produced, {}
