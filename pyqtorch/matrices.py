from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_REAL_DTYPE = torch.float64
DEFAULT_MATRIX_DTYPE = torch.cdouble

IMAT = torch.eye(2, dtype=DEFAULT_MATRIX_DTYPE)
XMAT = torch.tensor([[0, 1], [1, 0]], dtype=DEFAULT_MATRIX_DTYPE)
YMAT = torch.tensor([[0, -1j], [1j, 0]], dtype=DEFAULT_MATRIX_DTYPE)
ZMAT = torch.tensor([[1, 0], [0, -1]], dtype=DEFAULT_MATRIX_DTYPE)
SMAT = torch.tensor([[1, 0], [0, 1j]], dtype=DEFAULT_MATRIX_DTYPE)
SDAGGERMAT = torch.tensor([[1, 0], [0, -1j]], dtype=DEFAULT_MATRIX_DTYPE)
TMAT = torch.tensor(
    [
        [1, 0],
        [0, torch.exp(torch.tensor(1.0j * torch.pi / 4, dtype=DEFAULT_MATRIX_DTYPE))],
    ],
    dtype=DEFAULT_MATRIX_DTYPE,
)
NMAT = torch.tensor([[0, 0], [0, 1]], dtype=DEFAULT_MATRIX_DTYPE)
IDIAG = torch.tensor([1, 1], dtype=DEFAULT_MATRIX_DTYPE)

SWAPMAT = torch.tensor(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=DEFAULT_MATRIX_DTYPE
)
CSWAPMAT = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=DEFAULT_MATRIX_DTYPE,
)
HMAT = (
    1
    / torch.sqrt(torch.tensor(2.0, dtype=DEFAULT_MATRIX_DTYPE))
    * torch.tensor([[1, 1], [1, -1]], dtype=DEFAULT_MATRIX_DTYPE)
)


def PROJMAT(ket: Tensor, bra: Tensor) -> Tensor:
    return torch.outer(ket, bra)


OPERATIONS_DICT = {
    "I": IMAT,
    "X": XMAT,
    "Y": YMAT,
    "Z": ZMAT,
    "S": SMAT,
    "SDAGGER": SDAGGERMAT,
    "T": TMAT,
    "N": NMAT,
    "PROJ": PROJMAT,
    "H": HMAT,
    "SWAP": SWAPMAT,
    "CSWAP": CSWAPMAT,
}


def parametric_unitary(
    theta: torch.Tensor,
    P: torch.Tensor,
    identity_mat: torch.Tensor,
    batch_size: int,
    diagonal: bool = False,
    a: float = 0.5,  # noqa: E741
) -> torch.Tensor:
    """Compute the exponentiation of a Pauli matrix :math:`P`

    The exponentiation is given by:
    :math:`exp(-i a \\theta P ) = I cos(r \\theta) - i a P sin(r \\theta) / r`

    where :math:`a` is a prefactor
    and :math:`r = a * sg / 2`, :math:`sg` corresponding to the spectral gap.

    Here, we assume :math:`sg = 2`

    Args:
        theta (torch.Tensor): Parameter values.
        P (torch.Tensor): Pauli matrix to exponentiate.
        identity_mat (torch.Tensor): Identity matrix
        batch_size (int): Batch size of parameters.
        diagonal (bool): if dealing with diagonal operation.
        a (float): Prefactor.

    Returns:
        torch.Tensor: The exponentiation of P
    """
    cos_t = torch.cos(theta * a).unsqueeze(0)
    sin_t = torch.sin(theta * a).unsqueeze(0)
    batch_imat = identity_mat.unsqueeze(-1)
    batch_operation_mat = P.unsqueeze(-1)
    if not diagonal:
        cos_t = cos_t.unsqueeze(1)
        sin_t = sin_t.unsqueeze(1)
        cos_t = cos_t.repeat((2, 2, 1))
        sin_t = sin_t.repeat((2, 2, 1))

        batch_imat = batch_imat.repeat(1, 1, batch_size)
        batch_operation_mat = batch_operation_mat.repeat(1, 1, batch_size)
    else:
        batch_imat = batch_imat.repeat(1, batch_size)
        batch_operation_mat = batch_operation_mat.repeat(1, batch_size)

    return cos_t * batch_imat - 1j * sin_t * batch_operation_mat


def _dagger(matrices: torch.Tensor, diagonal: bool = False) -> torch.Tensor:
    return (
        torch.permute(matrices.conj(), (1, 0, 2)) if not diagonal else matrices.conj()
    )


def _jacobian(
    theta: torch.Tensor,
    P: torch.Tensor,
    identity_mat: torch.Tensor,
    batch_size: int,
    diagonal: bool = False,
) -> torch.Tensor:
    cos_t = torch.cos(theta / 2).unsqueeze(0)
    sin_t = torch.sin(theta / 2).unsqueeze(0)
    batch_imat = identity_mat.unsqueeze(-1)
    batch_operation_mat = P.unsqueeze(-1)
    if not diagonal:
        cos_t = cos_t.unsqueeze(1)
        sin_t = sin_t.unsqueeze(1)
        cos_t = cos_t.repeat((2, 2, 1))
        sin_t = sin_t.repeat((2, 2, 1))

        batch_imat = batch_imat.repeat(1, 1, batch_size)
        batch_operation_mat = batch_operation_mat.repeat(1, 1, batch_size)
    else:
        batch_imat = batch_imat.repeat(1, batch_size)
        batch_operation_mat = batch_operation_mat.repeat(1, batch_size)

    return -1 / 2 * (sin_t * batch_imat + 1j * cos_t * batch_operation_mat)


def controlled(
    operation: torch.Tensor,
    batch_size: int,
    n_control_qubits: int = 1,
    diagonal: bool = False,
) -> torch.Tensor:
    if diagonal:
        _controlled: torch.Tensor = (
            torch.ones(
                2 ** (n_control_qubits + 1),
                dtype=operation.dtype,
                device=operation.device,
            )
            .unsqueeze(1)
            .repeat(1, batch_size)
        )
        _controlled[2 ** (n_control_qubits + 1) - 2 :, :] = operation
    else:
        _controlled = (
            torch.eye(
                2 ** (n_control_qubits + 1),
                dtype=operation.dtype,
                device=operation.device,
            )
            .unsqueeze(2)
            .repeat(1, 1, batch_size)
        )
        _controlled[
            2 ** (n_control_qubits + 1) - 2 :, 2 ** (n_control_qubits + 1) - 2 :, :
        ] = operation
    return _controlled


COMPLEX_TO_REAL_DTYPES = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}


def add_batch_dim(operator: Tensor, batch_size: int = 1) -> Tensor:
    """In case we have a sequence of batched parametric gates mixed with primitive gates,
    we adjust the batch_dim of the primitive gates to match."""
    return (
        operator.repeat(1, 1, batch_size)
        if operator.shape != (2, 2, batch_size)
        else operator
    )
