from __future__ import annotations

from enum import Enum
from typing import Sequence

import torch

from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE

State = torch.Tensor
Operator = torch.Tensor


def overlap(bra: torch.Tensor, ket: torch.Tensor) -> torch.Tensor:
    n_qubits = len(bra.size()) - 1
    bra = bra.reshape((2**n_qubits, bra.size(-1)))
    ket = ket.reshape((2**n_qubits, ket.size(-1)))
    return torch.einsum("ib,ib->b", bra.conj(), ket).real


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
    Which Differentiation engine to use.

    Options: Automatic Differentiation -  Using torch.autograd.
             Adjoint Differentiation   - An implementation of "Efficient calculation of gradients
                                       in classical simulations of variational quantum algorithms",
                                       Jones & Gacon, 2020
    """

    AD = "ad"
    ADJOINT = "adjoint"


def is_normalized(state: torch.Tensor, atol: float = 1e-14) -> bool:
    n_qubits = len(state.size()) - 1
    batch_size = state.size()[-1]
    state = state.reshape((2**n_qubits, batch_size))
    sum_probs = (state.abs() ** 2).sum(dim=0)
    ones = torch.ones(batch_size, dtype=torch.double)
    return torch.allclose(sum_probs, ones, rtol=0.0, atol=atol)  # type: ignore[no-any-return]


def is_diag(H: torch.Tensor) -> bool:
    """
    Returns True if Hamiltonian H is diagonal.
    """
    return len(torch.abs(torch.triu(H, diagonal=1)).to_sparse().coalesce().values()) == 0


def product_state(
    bitstring: str, batch_size: int = 1, device: str | torch.device = "cpu"
) -> torch.Tensor:
    state = torch.zeros((2 ** len(bitstring), batch_size), dtype=DEFAULT_MATRIX_DTYPE)
    state[int(bitstring, 2)] = torch.tensor(1.0 + 0j, dtype=DEFAULT_MATRIX_DTYPE)
    return state.reshape([2] * len(bitstring) + [batch_size]).to(device=device)


def zero_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> torch.Tensor:
    return product_state("0" * n_qubits, batch_size, device)


def uniform_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> torch.Tensor:
    state = torch.ones((2**n_qubits, batch_size), dtype=dtype, device=device)
    state = state / torch.sqrt(torch.tensor(2**n_qubits))
    state = state.reshape([2] * n_qubits + [batch_size])
    return state.to(device=device)


def random_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = DEFAULT_MATRIX_DTYPE,
) -> torch.Tensor:
    def _normalize(wf: torch.Tensor) -> torch.Tensor:
        return wf / torch.sqrt((wf.abs() ** 2).sum())

    def _rand(n_qubits: int) -> torch.Tensor:
        N = 2**n_qubits
        x = -torch.log(torch.rand(N))
        sumx = torch.sum(x)
        phases = torch.rand(N) * 2.0 * torch.pi
        return _normalize(
            (torch.sqrt(x / sumx) * torch.exp(1j * phases)).reshape(N, 1).type(dtype).to(device)
        )

    _state = torch.concat(tuple(_rand(n_qubits) for _ in range(batch_size)), dim=1)
    return _state.reshape([2] * n_qubits + [batch_size]).to(device=device)


def param_dict(keys: Sequence[str], values: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: val for key, val in zip(keys, values)}
