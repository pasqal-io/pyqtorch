from __future__ import annotations

import time
from typing import Any, Callable, no_type_check

import torch

import pyqtorch.modules as pyq
from pyqtorch.core.batched_operation import batchedRX, batchedRY, batchedRZ

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.cdouble


def speed_test_module_circuit(batch_size: int = 10, n_qubits: int = 8, n_epochs: int = 100) -> None:
    ops = (
        [pyq.RX([i], n_qubits) for i in range(n_qubits)]
        + [pyq.RY([i], n_qubits) for i in range(n_qubits)]
        + [pyq.RZ([i], n_qubits) for i in range(n_qubits)]
    )

    circ = pyq.QuantumCircuit(n_qubits, ops).to(device=DEVICE, dtype=DTYPE)
    state = pyq.random_state(n_qubits, batch_size)
    phi = torch.rand(batch_size, device=DEVICE, dtype=DTYPE, requires_grad=True)
    for i in range(n_epochs):
        state = circ(state, phi)


def speed_test_func_circuit(batch_size: int = 10, n_qubits: int = 8, n_epochs: int = 100) -> None:
    ops = [batchedRX, batchedRY, batchedRZ]

    # circ = pyq.QuantumCircuit(n_qubits, ops).to(device=DEVICE, dtype=DTYPE)
    state = pyq.random_state(n_qubits, batch_size)
    phi = torch.rand(batch_size, device=DEVICE, dtype=DTYPE, requires_grad=True)
    for _ in range(n_epochs):
        for op in ops:
            # state = circ(state, phi)
            state = op(phi, state, [i for i in range(n_qubits)], n_qubits)


@no_type_check
def timeit(func: Callable) -> Any:
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper


@no_type_check
@timeit
def time_fn():
    speed_test_module_circuit()


if __name__ == "__main__":
    time_fn()
