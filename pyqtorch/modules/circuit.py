from __future__ import annotations

import torch
from torch.nn import Module


def zero_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    state = torch.zeros((2**n_qubits, batch_size), dtype=dtype, device=device)
    state[0] = 1
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


def uniform_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    state = torch.ones((2**n_qubits, batch_size), dtype=dtype, device=device)
    state = state / torch.sqrt(torch.tensor(2**n_qubits))
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


class QuantumCircuit(Module):
    def __init__(self, n_qubits: int, operations: list):
        super().__init__()
        self.n_qubits = n_qubits
        self.operations = torch.nn.ModuleList(operations)

    def forward(self, thetas: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        for op in self.operations:
            state = op(thetas, state)
        return state

    @property
    def _device(self) -> torch.device:
        devices = set(p.device for p in self.parameters())
        if len(devices) == 0:
            return torch.device("cpu")
        elif len(devices) == 1:
            return devices.pop()
        else:
            raise ValueError("only one device supported.")

    def init_state(self, batch_size: int) -> torch.Tensor:
        return zero_state(self.n_qubits, batch_size, device=self._device)
