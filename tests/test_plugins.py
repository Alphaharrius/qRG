from typing import Any

import pytest

from qrg.diagnostics import diagnostic, run
from qrg.diagnostics import names as diagnostic_names
from qrg.diagnostics import unregister as unregister_diagnostic
from qrg.seeds import make, seed
from qrg.seeds import names as seed_names
from qrg.seeds import unregister as unregister_seed
from qrg.tree import Node


def test_seed_registry_calls_the_named_recipe() -> None:
    @seed("unit-constant")
    def constant(scale: int) -> int:
        return scale

    try:
        assert make("unit-constant", scale=3) == 3
        assert "unit-constant" in seed_names()
        with pytest.raises(ValueError, match="already registered"):
            seed("unit-constant")(constant)
    finally:
        unregister_seed("unit-constant")

    with pytest.raises(ValueError, match="not registered"):
        make("unit-constant")


def test_diagnostic_records_its_result_on_the_node() -> None:
    @diagnostic("unit-count")
    def count(node: Node, *, offset: int = 0) -> int:
        return len(node) + offset

    root = Node.new(name="root", operator=_operator())
    try:
        assert run(root, "unit-count", offset=2) == 3
        assert root.info["unit-count"] == 3
        assert "unit-count" in diagnostic_names()
    finally:
        unregister_diagnostic("unit-count")


def _operator() -> Any:
    class _Tensor:
        pass

    return _Tensor()
