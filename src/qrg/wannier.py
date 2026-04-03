import warnings
from typing import Any, cast

from qten.geometries.fourier import fourier_transform
from qten.linalg.decompose import svd
from qten.linalg.tensors import Tensor
from qten.symbolics.hilbert_space import HilbertSpace
from qten.symbolics.state_space import MomentumSpace


def wannierize_k(
    eigenvectors: Tensor[Any], seeds: Tensor[Any], svd_threshold: float = 1e-1
) -> Tensor[Any]:
    """
    Perform projective wannierization on the target bands with the seeding states in momentum space.

    Parameters
    ----------
    eigenvectors : Tensor
        Target bands/eigenvectors. Expected shape `(MomentumSpace, HilbertSpace, IndexSpace)`.
    seeds : Tensor
        Seed states in momentum space. Expected shape `(MomentumSpace, HilbertSpace, IndexSpace)`.
    svd_threshold : float
        Warn if the minimum singular value drops below this, indicating linearly dependent seeds
        or poor overlap with target bands.

    Returns
    -------
    Tensor
        Wannierized states with shape `(MomentumSpace, HilbertSpace, IndexSpace)`.
    """
    if eigenvectors.rank() != 3 or seeds.rank() != 3:
        raise ValueError("Both eigenvectors and seeds must be rank-3 Tensors.")

    # 1. Compute the overlap matrix for each momentum sector
    # P_k = \psi_k^\dagger S_k
    # Resulting shape: (MomentumSpace, IndexSpace_bands, IndexSpace_seeds)
    overlap = eigenvectors.h(-2, -1) @ seeds

    # 2. Perform SVD on the overlap matrix
    U, S, Vh = svd(overlap)

    # Check for linear dependence / poor projection
    min_svd_val = S.data.min().item()
    if min_svd_val < svd_threshold:
        warnings.warn(
            f"Precarious wannier projection with minimum svd value of {min_svd_val:.4g}",
            UserWarning,
            stacklevel=2,
        )

    # 3. Construct the unitary transformation matrix
    # M_k = U_k V_k^\dagger
    unitary = U @ Vh

    # 4. Rotate the target bands into the Wannier gauge
    # W_k = \psi_k M_k
    wannier_states = eigenvectors @ unitary

    return cast(Tensor[Any], wannier_states)


def wannierize_r(
    eigenvectors: Tensor[Any], seeds: Tensor[Any], svd_threshold: float = 1e-1
) -> Tensor[Any]:
    """
    Perform projective wannierization using real-space localized seed states.

    Parameters
    ----------
    eigenvectors : Tensor
        Target bands with shape `(MomentumSpace, HilbertSpace, IndexSpace)`.
    seeds : Tensor
        Seed states localized in real space with shape `(HilbertSpace_local, IndexSpace)`.
    svd_threshold : float
        SVD warning threshold.

    Returns
    -------
    Tensor
        Wannierized states in momentum space.
    """
    if not isinstance(eigenvectors.dims[0], MomentumSpace):
        raise TypeError("The first dimension of the eigenvectors must be a MomentumSpace.")

    kspace = eigenvectors.dims[0]
    outspace = eigenvectors.dims[1]
    inspace_local = seeds.dims[0]
    if not isinstance(outspace, HilbertSpace) or not isinstance(inspace_local, HilbertSpace):
        raise TypeError(
            "The second dimension of eigenvectors and first dimension "
            "of seeds must be HilbertSpace."
        )

    # Perform Fourier transform on local seeds to move them to momentum space
    # f shape: (MomentumSpace, HilbertSpace_out, HilbertSpace_in_local)
    f = fourier_transform(kspace, outspace, inspace_local, device=eigenvectors.device)

    # Map the seeds to crystal momentum seeds
    # f @ local_seeds -> (MomentumSpace, HilbertSpace_out, IndexSpace)
    crystal_seeds = f @ seeds

    return wannierize_k(eigenvectors, crystal_seeds, svd_threshold)


def projective_wannierization(
    eigenvectors: Tensor[Any], seeds: Tensor[Any], svd_threshold: float = 1e-1
) -> Tensor[Any]:
    """
    Perform projective wannierization with automatic seed-space dispatch.

    Parameters
    ----------
    eigenvectors : Tensor
        Target bands with shape `(MomentumSpace, HilbertSpace, IndexSpace)`.
    seeds : Tensor
        Either crystal-momentum seeds `(MomentumSpace, HilbertSpace, IndexSpace)`
        or local real-space seeds `(HilbertSpace_local, IndexSpace)`.
    svd_threshold : float
        SVD warning threshold.

    Returns
    -------
    Tensor
        Wannierized states in momentum space.
    """
    if seeds.rank() == 3:
        if not isinstance(seeds.dims[0], MomentumSpace):
            raise TypeError("Rank-3 seeds must have MomentumSpace as the first dimension.")
        return wannierize_k(eigenvectors=eigenvectors, seeds=seeds, svd_threshold=svd_threshold)

    if seeds.rank() == 2:
        if not isinstance(seeds.dims[0], HilbertSpace):
            raise TypeError("Rank-2 seeds must have HilbertSpace as the first dimension.")
        return wannierize_r(eigenvectors=eigenvectors, seeds=seeds, svd_threshold=svd_threshold)

    raise ValueError("Seeds must be rank-2 (local seeds) or rank-3 (momentum seeds).")
