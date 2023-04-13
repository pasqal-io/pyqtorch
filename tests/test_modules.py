from __future__ import annotations

import pytest
import torch

import pyqtorch.core.batched_operation as func_pyq
import pyqtorch.modules as pyq


@pytest.mark.parametrize("gate", ["RX", "RY", "RZ"])
def test_gates(gate: str) -> None:
    dtype = torch.cdouble
    device = "cpu"
    batch_size = 100
    qubits = [0]
    n_qubits = 2

    state = pyq.zero_state(n_qubits, batch_size=batch_size, device=device, dtype=dtype)
    phi = torch.rand(batch_size, device=device, dtype=dtype)
    thetas = {"phi": phi}

    Op = getattr(pyq, gate)
    FuncOp = getattr(func_pyq, f"batched{gate}")

    func_out = FuncOp(phi, state, qubits, n_qubits)
    op = Op(qubits, n_qubits, "phi").to(device=device, dtype=dtype)
    mod_out = op(thetas, state)

    assert torch.allclose(func_out, mod_out)


def test_circuit() -> None:
    device = "cpu"
    dtype = torch.cdouble
    batch_size = 5

    ops = [pyq.X([0], 2), pyq.X([1], 2), pyq.RX([1], 2, "phi"), pyq.CNOT([0, 1], 2)]
    circ = pyq.QuantumCircuit(2, ops).to(device=device, dtype=dtype)

    state = circ.init_state(batch_size)
    phi = torch.rand(batch_size, device=device, dtype=dtype)
    thetas = {"phi": phi}

    assert circ(thetas, state).size() == (2, 2, 5)
