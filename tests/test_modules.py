from __future__ import annotations

import pytest
import torch

import pyqtorch.core.batched_operation as func_pyq
import pyqtorch.modules as pyq


@pytest.fixture
def simple_circ(n_qubits: int = 2) -> pyq.QuantumCircuit:
    ops = [
        pyq.RX([0], n_qubits, "theta_0"),
        pyq.RY([1], n_qubits, "theta_1"),
        pyq.RX([1], n_qubits, "theta_2"),
        pyq.CNOT([0, 1], n_qubits),
    ]
    return pyq.QuantumCircuit(n_qubits, ops)


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


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_circuit(batch_size: int) -> None:
    n_qubits = 2
    device = "cpu"
    dtype = torch.cdouble

    ops = [
        pyq.X([0], n_qubits),
        pyq.X([1], n_qubits),
        pyq.RX([1], n_qubits, "phi"),
        pyq.CNOT([0, 1], n_qubits),
    ]
    circ = pyq.QuantumCircuit(n_qubits, ops).to(device=device, dtype=dtype)

    state = circ.init_state(batch_size)
    phi = torch.rand(batch_size, device=device, dtype=dtype, requires_grad=True)
    thetas = {"phi": phi}

    assert circ(thetas, state).size() == (2, 2, batch_size)

    state = pyq.zero_state(n_qubits, batch_size=batch_size, device=device, dtype=dtype)

    res = circ.forward(thetas, state)
    assert not torch.all(torch.isnan(res))

    # g = torch.autograd.grad(circ, thetas)
    dres_theta = torch.autograd.grad(res, phi, torch.ones_like(res), create_graph=True)[0]
    assert not torch.all(torch.isnan(dres_theta))


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_U_gate(batch_size: int) -> None:
    n_qubits = 1
    u = pyq.U([0], n_qubits, ["phi", "theta", "omega"])
    d = {
        "phi": torch.rand(batch_size),
        "theta": torch.rand(batch_size),
        "omega": torch.rand(batch_size),
    }
    state = pyq.zero_state(n_qubits, batch_size=batch_size, device="cpu", dtype=torch.cdouble)
    assert not torch.all(torch.isnan(u.forward(d, state)))
