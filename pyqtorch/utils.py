from __future__ import annotations

import logging
from enum import Enum
from logging import getLogger
from typing import Sequence

import torch
from torch import Tensor

from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, DEFAULT_REAL_DTYPE

State = Tensor
Operator = Tensor

ATOL = 1e-06
RTOL = 0.0
GRADCHECK_ATOL = 1e-06

logger = getLogger(__name__)


def inner_prod(bra: Tensor, ket: Tensor) -> Tensor:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Inner prod calculation")
        torch.cuda.nvtx.range_push("inner_prod")

    n_qubits = len(bra.size()) - 1
    bra = bra.reshape((2**n_qubits, bra.size(-1)))
    ket = ket.reshape((2**n_qubits, ket.size(-1)))
    res = torch.einsum("ib,ib->b", bra.conj(), ket)
    if logger.isEnabledFor(logging.DEBUG):
        torch.cuda.nvtx.range_pop()
        logger.debug("Inner prod complete")
    return res


def overlap(bra: Tensor, ket: Tensor) -> Tensor:
    return torch.pow(inner_prod(bra, ket).real, 2)


class StrEnum(str, Enum):
    def __str__(self) -> str:
        """Used when dumping enum fields in a schema."""
        ret: str = self.value
        return ret

    @classmethod
    def list(cls) -> list[str]:
        return list(map(lambda c: c.value, cls))  # type: ignore


class DiffMode(StrEnum):
    """
    Which Differentiation method to use.

    Options: Automatic Differentiation -  Using torch.autograd.
             Adjoint Differentiation   - An implementation of "Efficient calculation of gradients
                                       in classical simulations of variational quantum algorithms",
                                       Jones & Gacon, 2020
    """

    AD = "ad"
    ADJOINT = "adjoint"


def is_normalized(state: Tensor, atol: float = ATOL) -> bool:
    n_qubits = len(state.size()) - 1
    batch_size = state.size()[-1]
    state = state.reshape((2**n_qubits, batch_size))
    sum_probs = (state.abs() ** 2).sum(dim=0)
    ones = torch.ones(batch_size, dtype=DEFAULT_REAL_DTYPE)
    return torch.allclose(sum_probs, ones, rtol=RTOL, atol=atol)  # type: ignore[no-any-return]


def is_diag(H: Tensor) -> bool:
    """
    Returns True if Hamiltonian H is diagonal.
    """
    return (
        len(torch.abs(torch.triu(H, diagonal=1)).to_sparse().coalesce().values()) == 0
    )


def product_state(
    bitstring: str,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> Tensor:
    state = torch.zeros((2 ** len(bitstring), batch_size), dtype=dtype)
    state[int(bitstring, 2)] = torch.tensor(1.0 + 0j, dtype=dtype)
    return state.reshape([2] * len(bitstring) + [batch_size]).to(device=device)


def zero_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> Tensor:
    return product_state("0" * n_qubits, batch_size, dtype=dtype, device=device)


def uniform_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> Tensor:
    state = torch.ones((2**n_qubits, batch_size), dtype=dtype, device=device)
    state = state / torch.sqrt(torch.tensor(2**n_qubits))
    return state.reshape([2] * n_qubits + [batch_size])


def random_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> Tensor:
    def _normalize(wf: Tensor) -> Tensor:
        return wf / torch.sqrt((wf.abs() ** 2).sum())

    def _rand(n_qubits: int) -> Tensor:
        N = 2**n_qubits
        x = -torch.log(torch.rand(N))
        sumx = torch.sum(x)
        phases = torch.rand(N) * 2.0 * torch.pi
        return _normalize(
            (torch.sqrt(x / sumx) * torch.exp(1j * phases))
            .reshape(N, 1)
            .type(dtype)
            .to(device)
        )

    state = torch.concat(tuple(_rand(n_qubits) for _ in range(batch_size)), dim=1)
    return state.reshape([2] * n_qubits + [batch_size]).to(device=device)


def param_dict(keys: Sequence[str], values: Sequence[Tensor]) -> dict[str, Tensor]:
    return {key: val for key, val in zip(keys, values)}


def batch_first(operator: Tensor) -> Tensor:
    """
    Permute the operator's batch dimension on first dimension.

    Args:
        operator (Tensor): Operator in size [2**n_qubits, 2**n_qubits,batch_size].

    Returns:
        Tensor: Operator in size [batch_size, 2**n_qubits, 2**n_qubits].
    """
    batch_first_perm = (2, 0, 1)
    return torch.permute(operator, batch_first_perm)


def batch_last(operator: Tensor) -> Tensor:
    """
    Permute the operator's batch dimension on last dimension.

    Args:
        operator (Tensor): Operator in size [batch_size,2**n_qubits, 2**n_qubits].

    Returns:
        Tensor: Operator in size [2**n_qubits, 2**n_qubits,batch_size].
    """
    undo_perm = (1, 2, 0)
    return torch.permute(operator, undo_perm)


def density_mat(state: Tensor) -> Tensor:
    """
    Computes the density matrix from a pure state vector.

    Args:
        state (Tensor): The pure state vector :math:`|\\psi\\rangle`.

    Returns:
        Tensor: The density matrix :math:`\\rho = |\psi \\rangle \\langle\\psi|`.
    """
    n_qubits = len(state.size()) - 1
    batch_dim = state.dim() - 1
    batch_size = state.shape[-1]
    batch_first_perm = [batch_dim] + list(range(batch_dim))
    state = torch.permute(state, batch_first_perm).reshape(batch_size, 2**n_qubits)
    return batch_last(torch.einsum("bi,bj->bij", (state, state.conj())))


def operator_kron(op1: Tensor, op2: Tensor) -> Tensor:
    """
    Compute the Kronecker product of two operators.

    Prevents errors related to the shape of the operators
    [2**n_qubits, 2**n_qubits, batch_size] when simply using `torch.kron()`.
    Use of `.contiguous()` to avoid errors related to the `torch.kron()` of a transposing tensor

    Args:
        op1 (Tensor): The first input tensor.
        op2 (Tensor): The second input tensor.

    Returns:
        Tensor: The resulting tensor after applying the Kronecker product
    """
    batch_size_1, batch_size_2 = op1.size(2), op2.size(2)
    if batch_size_1 > batch_size_2:
        op2 = op2.repeat(1, 1, batch_size_1)[:, :, :batch_size_1]
    elif batch_size_2 > batch_size_1:
        op1 = op1.repeat(1, 1, batch_size_2)[:, :, :batch_size_2]
    kron_product = torch.einsum(
        "bik,bjl->bijkl", batch_first(op1).contiguous(), batch_first(op2).contiguous()
    )
    return batch_last(
        kron_product.reshape(
            op1.size(2), op1.size(1) * op2.size(1), op1.size(0) * op2.size(0)
        )
    )


def promote_operator(operator: Tensor, target: int, n_qubits: int) -> Tensor:
    from pyqtorch.primitive import I

    """
    Promotes `operator` to the size of the circuit (number of qubits and batch).
    Targeting the first qubit implies target = 0, so target > n_qubits - 1.

    Args:
        operator (Tensor): The operator tensor to be promoted.
        target (int): The index of the target qubit to which the operator is applied.
            Targeting the first qubit implies target = 0, so target > n_qubits - 1.
        n_qubits (int): Number of qubits in the circuit.

    Returns:
        Tensor: The promoted operator tensor.

    Raises:
        ValueError: If `target` is outside the valid range of qubits.
    """
    if target > n_qubits - 1:
        raise ValueError(
            "The target must be a valid qubit index within the circuit's range."
        )
    qubits = torch.arange(0, n_qubits)
    qubits = qubits[qubits != target]
    for qubit in qubits:
        operator = torch.where(
            target > qubit,
            operator_kron(I(target).unitary(), operator),
            operator_kron(operator, I(target).unitary()),
        )
    return operator


def operator_to_sparse_diagonal(operator: Tensor) -> Tensor:
    operator = torch.diag(operator)
    indices, values, size = (
        torch.nonzero(operator),
        operator[operator != 0],
        len(operator),
    )
    indices = torch.stack((indices.flatten(), indices.flatten()))
    return torch.sparse_coo_tensor(indices, values, (size, size))
