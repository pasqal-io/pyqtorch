from __future__ import annotations

import logging
from enum import Enum
from logging import getLogger
from math import log2
from string import ascii_uppercase as ABC
from typing import Sequence

import torch
from numpy import array
from numpy import ndarray as NDArray
from torch import Tensor

from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, DEFAULT_REAL_DTYPE

State = Tensor
Operator = Tensor

ATOL = 1e-06
RTOL = 0.0
GRADCHECK_ATOL = 1e-06
ABC_ARRAY: NDArray = array(list(ABC))

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
    return len(torch.abs(torch.triu(H, diagonal=1)).to_sparse().coalesce().values()) == 0


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
            (torch.sqrt(x / sumx) * torch.exp(1j * phases)).reshape(N, 1).type(dtype).to(device)
        )

    state = torch.concat(tuple(_rand(n_qubits) for _ in range(batch_size)), dim=1)
    return state.reshape([2] * n_qubits + [batch_size]).to(device=device)


def param_dict(keys: Sequence[str], values: Sequence[Tensor]) -> dict[str, Tensor]:
    return {key: val for key, val in zip(keys, values)}


class DensityMatrix(Tensor):
    pass


def density_mat(state: DensityMatrix) -> DensityMatrix:
    """
    Computes the density matrix from a pure state vector.

    Args:
        state (Tensor): The pure state vector :math:`|\\psi\\rangle`.

    Returns:
        Tensor: The density matrix :math:`\\rho = |\psi \\rangle \\langle\\psi|`.
    """
    n_qubits = len(state.size()) - 1
    batch_size = state.shape[-1]
    state = state.reshape(2**n_qubits, batch_size)
    return DensityMatrix(torch.einsum("ib,jb->ijb", (state, state.conj())))


def operator_kron(op1: Tensor, op2: Tensor) -> Tensor:
    """
    Computes the Kronecker product of two operators.

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
    kron_product = torch.einsum("ikb,jlb->ijklb", op1.contiguous(), op2.contiguous())
    return kron_product.reshape(op1.size(0) * op2.size(0), op1.size(1) * op2.size(1), op1.size(2))


def add_batch_dim(operator: Tensor, batch_size: int = 1) -> Tensor:
    """In case we have a sequence of batched parametric gates mixed with primitive gates,
    we adjust the batch_dim of the primitive gates to match."""
    return operator.repeat(1, 1, batch_size) if operator.shape != (2, 2, batch_size) else operator


def dm_partial_trace(rho: DensityMatrix, keep_indices: list[int]) -> DensityMatrix:
    """
    Computes the partial trace of a density matrix for a system of several qubits with batch size.
    This function also permutes the qubits according to the order specified in keep_indices.

    Args:
        rho (DensityMatrix) : Density matrix of shape [2**n_qubits, 2**n_qubits, batch_size].
        keep_indices (list[int]): Index of the qubit subsystems to keep.

    Returns:
        DensityMatrix: Reduced density matrix after the partial trace,
        of shape [2**n_keep, 2**n_keep, batch_size].
    """
    n_qubits = int(log2((rho.shape[0])))
    batch_size = rho.shape[2]
    rho = rho.reshape(([2] * n_qubits * 2 + [batch_size]))

    rho_subscripts = "".join(ABC_ARRAY[: n_qubits * 2 + 1])
    keep_subscripts = "".join([rho_subscripts[i] for i in keep_indices]) + "".join(
        [rho_subscripts[i + n_qubits] for i in keep_indices]
    )
    einsum_subscripts = rho_subscripts + "->" + keep_subscripts + rho_subscripts[n_qubits * 2]

    rho_reduced = torch.einsum(einsum_subscripts, rho)
    n_keep = len(keep_indices)
    return rho_reduced.reshape(2**n_keep, 2**n_keep, batch_size)  # type: ignore[no-any-return]


def operator_product(op1: Tensor, op2: Tensor) -> Tensor:
    """
    Computes the product of two operators.

    Args:
        op1 (Tensor): The first operator.
        op2 (Tensor): The second operator.
        target (int): The target qubit index.

    Returns:
        Tensor: The product of the two operators.

    Raises:
        ValueError: If the number of qubits of the input operators are not equal.
    """
    n_qubits_1 = int(log2(op1.size(1)))
    n_qubits_2 = int(log2(op2.size(1)))
    if n_qubits_1 != n_qubits_2:
        raise ValueError("The operators must have the same qubit number")
    batch_size_1 = op1.size(-1)
    batch_size_2 = op2.size(-1)
    if batch_size_1 > batch_size_2:
        op2 = op2.repeat(1, 1, batch_size_1)[:, :, :batch_size_1]
    elif batch_size_2 > batch_size_1:
        op1 = op1.repeat(1, 1, batch_size_2)[:, :, :batch_size_2]
    return torch.einsum("ijb,jkb->ikb", op1, op2)


def dm_kron(
    dm1: DensityMatrix, dm2: DensityMatrix, qubits_dm1: list[int], qubits_dm2: list[int]
) -> DensityMatrix:
    """
    Computes the Kronecker product of density matrices representing subsystems to reconstruct
    the density matrix of the global system.
    It allows flexibility in the order or indices of the subsystems provided.
    The order of qubits in qubits_dm1 and in qubits_dm2 determines how the subsystems are combined.

    Args:
        dm1 (DensityMatrix): The first input tensor.
        dm2 (DensityMatrix): The second input tensor.
        qubits_dm1 (list[int]): Indices of qubits for the first operator.
        qubits_dm2 (list[int]): Indices of qubits for the second operator.

    Returns:
        DensityMatrix: The density matrix of the global system.

    Raises:
        ValueError: If the batch sizes of the input operators are not equal.
    """
    batch_size_1 = dm1.size(-1)
    batch_size_2 = dm2.size(-1)
    if batch_size_1 != batch_size_2:
        raise ValueError("The batch sizes of the input operators must be equal.")
    n_qubits_1 = len(qubits_dm1)
    n_qubits_2 = len(qubits_dm2)
    total_qubits = sorted(qubits_dm1 + qubits_dm2)
    n_total_qubits = len(total_qubits)
    dm1 = dm1.reshape(([2] * n_qubits_1 * 2 + [batch_size_1]))
    dm2 = dm2.reshape(([2] * n_qubits_2 * 2 + [batch_size_1]))

    subscripts_dm1 = ABC_ARRAY[: n_qubits_1 * 2 + 1]
    subscripts_dm2 = [ABC_ARRAY[i + n_qubits_1 * 2 + 1] for i in range(n_qubits_2 * 2 + 1)]
    subscripts_dm2[-1] = subscripts_dm1[-1]
    subscripts_dm1, subscripts_dm2 = list(  # type: ignore[assignment]
        map(lambda e: "".join(list(e)), [subscripts_dm1, subscripts_dm2])
    )
    output_subscripts = [""] * n_total_qubits * 2 + [""]
    for idx, qubit in enumerate(total_qubits):
        if qubit in qubits_dm1:
            output_subscripts[idx] = subscripts_dm1[qubits_dm1.index(qubit)]
            output_subscripts[idx + n_total_qubits] = subscripts_dm1[
                qubits_dm1.index(qubit) + n_qubits_1
            ]
        else:
            output_subscripts[idx] = subscripts_dm2[qubits_dm2.index(qubit)]
            output_subscripts[idx + n_total_qubits] = subscripts_dm2[
                qubits_dm2.index(qubit) + n_qubits_2
            ]
    output_subscripts[-1] = subscripts_dm1[-1]

    einsum_notation = f"{subscripts_dm1},{subscripts_dm2}->" + "".join(output_subscripts)
    kron_product = torch.einsum(einsum_notation, dm1, dm2)
    return kron_product.reshape((2**n_total_qubits, 2**n_total_qubits, batch_size_1))  # type: ignore[no-any-return]
