from typing import Any

from qten.linalg.tensors import Tensor

def wannierize_k(
    eigenvectors: Tensor[Any], seeds: Tensor[Any], svd_threshold: float = 1e-1
) -> Tensor[Any]: ...
def wannierize_r(
    eigenvectors: Tensor[Any], seeds: Tensor[Any], svd_threshold: float = 1e-1
) -> Tensor[Any]: ...
def projective_wannierization(
    eigenvectors: Tensor[Any], seeds: Tensor[Any], svd_threshold: float = 1e-1
) -> Tensor[Any]: ...
