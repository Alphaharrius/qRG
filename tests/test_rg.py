from __future__ import annotations

from typing import Any

import pytest

from qrg.rg import lift, project, subspaces


class FakeTensor:
    def __init__(self, name: str) -> None:
        self.name = name

    def __matmul__(self, other: FakeTensor) -> FakeTensor:
        return FakeTensor(f"({self.name}@{other.name})")

    def h(self, *_axes: int) -> FakeTensor:
        return FakeTensor(f"{self.name}.h")


def test_project_and_lift_use_the_last_two_axes() -> None:
    operator = FakeTensor("H")
    transform = FakeTensor("W")

    assert cast_name(project(operator, transform)) == "((W.h@H)@W)"
    assert cast_name(lift(operator, transform)) == "((W@H)@W.h)"


def test_subspaces_projects_each_isometry_and_keeps_extra_records() -> None:
    branches = subspaces(
        FakeTensor("folded"),
        {"frozen": FakeTensor("Wf"), "courier": FakeTensor("Wc")},
        extra={"courier": {"seeds": FakeTensor("S")}},
        info={"frozen": {"overlap": 0.91}, "courier": {"overlap": 0.37}},
        kind="wannier",
    )

    assert cast_name(branches["courier"].data["H"]) == "((Wc.h@folded)@Wc)"
    assert cast_name(branches["courier"].data["seeds"]) == "S"
    assert branches["courier"].info["overlap"] == 0.37
    assert branches["frozen"].kind == "wannier"
    assert cast_name(branches["courier"].transform) == "Wc"


def test_subspaces_rejects_unknown_branches_and_a_replacement_operator() -> None:
    with pytest.raises(ValueError, match="do not match a branch"):
        subspaces(FakeTensor("H"), {"courier": FakeTensor("W")}, info={"frozen": {}})
    with pytest.raises(ValueError, match="must not include 'H'"):
        subspaces(
            FakeTensor("H"),
            {"courier": FakeTensor("W")},
            extra={"courier": {"H": FakeTensor("other")}},
        )


def cast_name(value: Any) -> str:
    return str(value.name)
