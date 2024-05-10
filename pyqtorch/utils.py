from __future__ import annotations

from enum import Enum
from typing import Sequence

import torch
from torch import Tensor

from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, DEFAULT_REAL_DTYPE

State = Tensor
Operator = Tensor

ATOL = 1e-06
RTOL = 0.0
GRADCHECK_ATOL = 1e-06


def inner_prod(bra: Tensor, ket: Tensor) -> Tensor:
    n_qubits = len(bra.size()) - 1
    bra = bra.reshape((2**n_qubits, bra.size(-1)))
    ket = ket.reshape((2**n_qubits, ket.size(-1)))
    return torch.einsum("ib,ib->b", bra.conj(), ket)


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
    undo_perm = (1, 2, 0)
    return torch.permute(torch.einsum("bi,bj->bij", (state, state.conj())), undo_perm)


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
        raise ValueError("The target must be a valid qubit index within the circuit's range.")
    qubits = torch.arange(0, n_qubits)
    qubits = qubits[qubits != target]
    for qubit in qubits:
        operator = torch.where(
            target > qubit,
            torch.kron(I(target).unitary(), operator.contiguous()),
            torch.kron(operator.contiguous(), I(target).unitary()),
        )
    return operator


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
