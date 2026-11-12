from __future__ import annotations

from typing import Any

import pytest
from sympy import ImmutableDenseMatrix

import qrg.strat as strat_module
from qrg import seeds
from qrg.strat import _lattice_dim, normal_blocking, proj_wannierization, upscale
from qrg.tree import Node


class FakeTensor:
    def __init__(self, name: str) -> None:
        self.name = name

    def __matmul__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(f"({self.name}@{other.name})")

    def h(self, *_axes: int) -> FakeTensor:
        return FakeTensor(f"{self.name}.h")


def test_normal_blocking_rejects_bad_factors() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        normal_blocking(0)
    with pytest.raises(ValueError, match="sequence of ints"):
        normal_blocking("2")  # type: ignore[arg-type]


def test_normal_blocking_folds_every_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strat_module, "_lattice_dim", lambda _operator: 2)
    seen: dict[str, Any] = {}

    def bandfold(transform: Any, tensor: Any, opt: str = "both") -> FakeTensor:
        seen["matrix"] = transform.M
        seen["opt"] = opt
        seen["tensor"] = tensor
        return FakeTensor("folded")

    monkeypatch.setattr("qten.bands.bandfold", bandfold)
    root = Node.new(name="root", operator=FakeTensor("H"))  # type: ignore[arg-type]
    child = root.add_strat("blocking", normal_blocking(2)).grow("blocking")

    assert cast_name(child.operator()) == "folded"
    assert child.step is not None
    assert child.step.kind == "blocking"
    assert child.step.transform is None
    assert child.step.params == {"factor": 2}
    assert child.info["factor"] == 2
    assert seen["opt"] == "both"
    assert seen["tensor"] is root.operator()
    assert seen["matrix"] == ImmutableDenseMatrix([[2, 0], [0, 2]])


def test_normal_blocking_checks_factor_length(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strat_module, "_lattice_dim", lambda _operator: 2)
    root = Node.new(name="root", operator=FakeTensor("H"))  # type: ignore[arg-type]
    root.add_strat("blocking", normal_blocking((2, 2, 1)))

    with pytest.raises(ValueError, match="length 3"):
        root.grow("blocking")


def test_upscale_interpolates_every_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strat_module, "_lattice_dim", lambda _operator: 2)
    seen: dict[str, Any] = {}

    def upsample(tensor: Any, scale: tuple[int, ...]) -> FakeTensor:
        seen["tensor"] = tensor
        seen["scale"] = scale
        return FakeTensor("upsampled")

    monkeypatch.setattr("qten.bands.upsample", upsample)
    root = Node.new(name="root", operator=FakeTensor("H"))  # type: ignore[arg-type]
    child = root.add_strat("upscale", upscale(2)).grow("upscale")

    assert cast_name(child.operator()) == "upsampled"
    assert child.step is not None
    assert child.step.kind == "upscale"
    assert child.step.transform is None
    assert child.info["factor"] == 2
    assert seen["tensor"] is root.operator()
    assert seen["scale"] == (2, 2)


def test_proj_wannierization_stores_both_isometries_and_continues_from_courier() -> None:
    @seeds.seed("unit-frozen")
    def frozen_seed(_operator: Any) -> FakeTensor:
        return FakeTensor("Wf")

    @seeds.seed("unit-courier")
    def courier_seed(_operator: Any, **_ignored: object) -> FakeTensor:
        return FakeTensor("Wc")

    root = Node.new(name="root", operator=FakeTensor("H"))  # type: ignore[arg-type]
    try:
        layer = root.add_strat(
            "wannier",
            proj_wannierization(frozen="unit-frozen", courier="unit-courier"),
        ).grow("wannier")
    finally:
        seeds.unregister("unit-frozen")
        seeds.unregister("unit-courier")

    assert cast_name(layer.operator()) == "((Wc.h@H)@Wc)"
    assert cast_name(layer["frozen"]) == "Wf"
    assert cast_name(layer["courier"]) == "Wc"
    assert layer.step is not None
    assert cast_name(layer.step.transform) == "Wc"
    assert layer.path() == "wannier"


def test_lattice_dim_reads_the_reciprocal_lattice() -> None:
    class _Lattice:
        dim = 3

    class _Momentum:
        def extract(self, _kind: Any) -> _Lattice:
            return _Lattice()

    class _Operator:
        dims = (_Momentum(),)

    assert _lattice_dim(_Operator()) == 3


def cast_name(value: Any) -> str:
    return str(value.name)
