from __future__ import annotations

import torch
from torch import Tensor

DEFAULT_MATRIX_DTYPE = torch.cdouble

IMAT = torch.eye(2, dtype=DEFAULT_MATRIX_DTYPE)
XMAT = torch.tensor([[0, 1], [1, 0]], dtype=DEFAULT_MATRIX_DTYPE)
YMAT = torch.tensor([[0, -1j], [1j, 0]], dtype=DEFAULT_MATRIX_DTYPE)
ZMAT = torch.tensor([[1, 0], [0, -1]], dtype=DEFAULT_MATRIX_DTYPE)
SMAT = torch.tensor([[1, 0], [0, 1j]], dtype=DEFAULT_MATRIX_DTYPE)
SDAGGERMAT = torch.tensor([[1, 0], [0, -1j]], dtype=DEFAULT_MATRIX_DTYPE)
TMAT = torch.tensor(
    [[1, 0], [0, torch.exp(torch.tensor(1.0j * torch.pi / 4, dtype=DEFAULT_MATRIX_DTYPE))]],
    dtype=DEFAULT_MATRIX_DTYPE,
)
NMAT = torch.tensor([[0, 0], [0, 1]], dtype=DEFAULT_MATRIX_DTYPE)
NDIAG = torch.tensor([0, 1], dtype=DEFAULT_MATRIX_DTYPE)
ZDIAG = torch.tensor([1, -1], dtype=DEFAULT_MATRIX_DTYPE)
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


def _unitary(
    theta: torch.Tensor, P: torch.Tensor, I: torch.Tensor, batch_size: int  # noqa: E741
) -> torch.Tensor:
    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    batch_imat = I.unsqueeze(2).repeat(1, 1, batch_size)
    batch_operation_mat = P.unsqueeze(2).repeat(1, 1, batch_size)

    return cos_t * batch_imat - 1j * sin_t * batch_operation_mat


def _dagger(matrices: torch.Tensor) -> torch.Tensor:  # noqa: E741
    return torch.permute(matrices.conj(), (1, 0, 2))


def _jacobian(
    theta: torch.Tensor, P: torch.Tensor, I: torch.Tensor, batch_size: int  # noqa: E741
) -> torch.Tensor:
    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    batch_imat = I.unsqueeze(2).repeat(1, 1, batch_size)
    batch_operation_mat = P.unsqueeze(2).repeat(1, 1, batch_size)

    return -1 / 2 * (sin_t * batch_imat + 1j * cos_t * batch_operation_mat)


def _controlled(unitary: torch.Tensor, batch_size: int, n_control_qubits: int = 1) -> torch.Tensor:
    _controlled: torch.Tensor = (
        torch.eye(2 ** (n_control_qubits + 1), dtype=DEFAULT_MATRIX_DTYPE)
        .unsqueeze(2)
        .repeat(1, 1, batch_size)
    )
    _controlled[2 ** (n_control_qubits + 1) - 2 :, 2 ** (n_control_qubits + 1) - 2 :, :] = unitary
    return _controlled
