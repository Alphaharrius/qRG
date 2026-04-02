from dataclasses import dataclass
from typing import Any

import pytest
import sympy as sy
import torch
from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.fourier import fourier_transform
from qten.geometries.spatials import Lattice, Offset
from qten.linalg.tensors import Tensor
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import IndexSpace, brillouin_zone
from sympy import ImmutableDenseMatrix

from qrg.wannier import projective_wannierization, wannier_projection


@dataclass(frozen=True)
class Orb:
    name: str


def _state(r: Offset[Any], orb: str = "s") -> U1Basis:
    return U1Basis(coef=sy.Integer(1), base=(r, Orb(orb)))


def _build_1d_spaces() -> tuple[Lattice, Any, HilbertSpace]:
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lattice.affine)
    r_half = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice.affine)
    bloch_space = HilbertSpace.new([_state(r0, "a"), _state(r_half, "b")])
    return lattice, k_space, bloch_space


def test_wannier_projection_matches_explicit_crystal_seed_pipeline() -> None:
    """Test local-seed projection matches explicit crystal-seed workflow."""
    # Minimal 1D lattice and Brillouin zone, similar to the notebook flow.
    _, k_space, bloch_space = _build_1d_spaces()
    local_space = bloch_space

    band_space = IndexSpace.linear(1)
    seed_space = IndexSpace.linear(1)

    # (K, B, I): one target band at each k with deterministic phase structure.
    eigenvectors = Tensor(
        data=torch.tensor(
            [
                [[2**-0.5], [2**-0.5]],
                [[2**-0.5], [-(2**-0.5)]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, bloch_space, band_space),
    )

    # (B_local, I): one local seed orbital.
    local_seeds = Tensor(
        data=torch.tensor([[1.0], [0.0]], dtype=torch.complex128),
        dims=(local_space, seed_space),
    )

    # Notebook-like pathway: local seeds -> Fourier seeds -> projective wannierization.
    crystal_seeds = fourier_transform(k_space, bloch_space, local_space) @ local_seeds
    expected = projective_wannierization(eigenvectors=eigenvectors, seeds=crystal_seeds)
    actual = wannier_projection(eigenvectors=eigenvectors, seeds=local_seeds)

    assert actual.dims == expected.dims
    assert torch.allclose(actual.data, expected.data)

    # Result should remain orthonormal within the selected band subspace.
    overlap = actual.h(-2, -1) @ actual
    assert torch.allclose(
        overlap.data,
        torch.ones((k_space.dim, 1, 1), dtype=torch.complex128),
    )


def test_projective_wannierization_rejects_non_rank3_tensors() -> None:
    """Test rank validation raises when inputs are not rank-3 tensors."""
    _, k_space, bloch_space = _build_1d_spaces()
    band_space = IndexSpace.linear(1)
    seed_space = IndexSpace.linear(1)

    eigenvectors_rank2 = Tensor(
        data=torch.tensor([[1.0], [0.0]], dtype=torch.complex128),
        dims=(bloch_space, band_space),
    )
    seeds_rank3 = Tensor(
        data=torch.ones((k_space.dim, bloch_space.dim, seed_space.dim), dtype=torch.complex128),
        dims=(k_space, bloch_space, seed_space),
    )
    with pytest.raises(ValueError, match="rank-3"):
        projective_wannierization(eigenvectors=eigenvectors_rank2, seeds=seeds_rank3)

    eigenvectors_rank3 = Tensor(
        data=torch.ones((k_space.dim, bloch_space.dim, band_space.dim), dtype=torch.complex128),
        dims=(k_space, bloch_space, band_space),
    )
    seeds_rank2 = Tensor(
        data=torch.tensor([[1.0], [0.0]], dtype=torch.complex128),
        dims=(bloch_space, seed_space),
    )
    with pytest.raises(ValueError, match="rank-3"):
        projective_wannierization(eigenvectors=eigenvectors_rank3, seeds=seeds_rank2)


def test_wannier_projection_rejects_non_momentum_first_dimension() -> None:
    """Test wannier_projection rejects non-MomentumSpace first tensor dim."""
    _, k_space, bloch_space = _build_1d_spaces()
    band_space = IndexSpace.linear(1)
    seed_space = IndexSpace.linear(1)

    bad_k_space = IndexSpace.linear(2)
    eigenvectors = Tensor(
        data=torch.ones((bad_k_space.dim, bloch_space.dim, band_space.dim), dtype=torch.complex128),
        dims=(bad_k_space, bloch_space, band_space),
    )
    local_seeds = Tensor(
        data=torch.tensor([[1.0], [0.0]], dtype=torch.complex128),
        dims=(bloch_space, seed_space),
    )
    with pytest.raises(TypeError, match="MomentumSpace"):
        wannier_projection(eigenvectors=eigenvectors, seeds=local_seeds)

    good_eigenvectors = Tensor(
        data=torch.ones((k_space.dim, bloch_space.dim, band_space.dim), dtype=torch.complex128),
        dims=(k_space, bloch_space, band_space),
    )
    wannier_projection(eigenvectors=good_eigenvectors, seeds=local_seeds)


def test_projective_wannierization_warns_on_poor_overlap() -> None:
    """Test poor seed-band overlap emits the precarious projection warning."""
    _, k_space, bloch_space = _build_1d_spaces()
    band_space = IndexSpace.linear(1)
    seed_space = IndexSpace.linear(1)

    # Build nearly orthogonal eigenvector/seed overlap to trigger warning.
    eigenvectors = Tensor(
        data=torch.tensor(
            [
                [[1.0], [0.0]],
                [[1.0], [0.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, bloch_space, band_space),
    )
    tiny = 1.0e-8
    seeds = Tensor(
        data=torch.tensor(
            [
                [[tiny], [1.0]],
                [[tiny], [1.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, bloch_space, seed_space),
    )

    with pytest.warns(UserWarning, match="Precarious wannier projection"):
        _ = projective_wannierization(
            eigenvectors=eigenvectors,
            seeds=seeds,
            svd_threshold=1.0e-3,
        )


def test_wannier_projection_projector_is_gauge_invariant() -> None:
    """Test projector is invariant between equivalent seed construction routes."""
    _, k_space, bloch_space = _build_1d_spaces()
    band_space = IndexSpace.linear(1)
    seed_space = IndexSpace.linear(1)

    eigenvectors = Tensor(
        data=torch.tensor(
            [
                [[2**-0.5], [2**-0.5]],
                [[2**-0.5], [-(2**-0.5)]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, bloch_space, band_space),
    )
    local_seeds = Tensor(
        data=torch.tensor([[0.0], [1.0]], dtype=torch.complex128),
        dims=(bloch_space, seed_space),
    )

    w_local = wannier_projection(eigenvectors=eigenvectors, seeds=local_seeds)
    w_crystal = projective_wannierization(
        eigenvectors=eigenvectors,
        seeds=fourier_transform(k_space, bloch_space, bloch_space) @ local_seeds,
    )

    p_local = w_local @ w_local.h(-2, -1)
    p_crystal = w_crystal @ w_crystal.h(-2, -1)
    assert torch.allclose(p_local.data, p_crystal.data)
