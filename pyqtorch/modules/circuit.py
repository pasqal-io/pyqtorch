from __future__ import annotations

import torch
from torch.nn import Module
from numpy.typing import ArrayLike


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

    def forward(self, thetas: dict[str, torch.Tensor], state: torch.Tensor):
        for op in self.operations:
            state = op(thetas, state)
        return state

    @property
    def device(self):
        devices = set(op.device for op in self.operations)
        if len(devices) == 1:
            return devices.pop()
        else:
            raise ValueError("only one device supported.")

    def init_state(self, batch_size):
        return zero_state(self.n_qubits, batch_size, device=self.device)


if __name__ == "__main__":
    from pyqtorch.core.batched_operation import batchedRX
    import time

    # print(OPERATIONS_DICT["I"])
    # g = OPERATIONS_DICT["I"].to(device="mps", dtype=torch.cfloat)
    # print(g + g)

    dtype = torch.cdouble
    device = "cpu"
    batch_size = 1_000
    state = uniform_state(2, batch_size=batch_size, device=device, dtype=dtype)
    phi = torch.rand(batch_size, device=device, dtype=dtype)
    thetas = {"phi": phi}

    qubits = [0]
    n_qubits = 2
    iters = 1_000

    t1 = time.time()
    for _ in range(iters):
        batchedRX(phi, state, qubits, n_qubits)
    print(f"{time.time()-t1}")

    gate = RX(qubits, n_qubits, "phi").to(device=device, dtype=dtype)
    # gate = torch.compile(gate)
    t1 = time.time()
    for _ in range(iters):
        gate(thetas, state)
    print(f"{time.time()-t1}")
    print(state)

    ops = [X([0], 2), X([1], 2), RX([1], 2, "phi"), CNOT([0, 1], 2)]
    ops = [RX([0], 2, "phi")]
    circ = QuantumCircuit(2, ops).to(device=device, dtype=dtype)

    circ(thetas, state)
    t1 = time.time()
    for _ in range(iters):
        circ(thetas, state)
    print(f"{time.time()-t1}")
    print(state)

    print()
