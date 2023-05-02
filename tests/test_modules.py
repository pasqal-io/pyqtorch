from __future__ import annotations

import pytest
import torch

import pyqtorch.core as func_pyq
import pyqtorch.modules as pyq


@pytest.mark.parametrize("gate", ["X", "Y", "Z"])
def test_constant_gates(gate: str) -> None:
    dtype = torch.cdouble
    device = "cpu"
    batch_size = 100
    qubits = [0]
    n_qubits = 2

    state = pyq.zero_state(n_qubits, batch_size=batch_size, device=device, dtype=dtype)
    phi = torch.rand(batch_size, device=device, dtype=dtype)

    Op = getattr(pyq, gate)
    FuncOp = getattr(func_pyq.operation, gate)

    func_out = FuncOp(state, qubits, n_qubits)
    op = Op(qubits, n_qubits).to(device=device, dtype=dtype)
    mod_out = op(state)

    assert torch.allclose(func_out, mod_out)


@pytest.mark.parametrize("gate", ["RX", "RY", "RZ"])
def test_parametrized_gates(gate: str) -> None:
    dtype = torch.cdouble
    device = "cpu"
    batch_size = 100
    qubits = [0]
    n_qubits = 2

    state = pyq.zero_state(n_qubits, batch_size=batch_size, device=device, dtype=dtype)
    phi = torch.rand(batch_size, device=device, dtype=dtype)

    Op = getattr(pyq, gate)
    FuncOp = getattr(func_pyq.batched_operation, f"batched{gate}")

    func_out = FuncOp(phi, state, qubits, n_qubits)
    op = Op(qubits, n_qubits).to(device=device, dtype=dtype)
    mod_out = op(state, phi)

    assert torch.allclose(func_out, mod_out)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_circuit(batch_size: int) -> None:
    n_qubits = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.cdouble

    ops = [
        pyq.X([0], n_qubits),
        pyq.X([1], n_qubits),
        pyq.RX([1], n_qubits),
        pyq.CNOT([0, 1], n_qubits),
    ]
    circ = pyq.QuantumCircuit(n_qubits, ops).to(device=device, dtype=dtype)

    state = circ.init_state(batch_size)
    phi = torch.rand(batch_size, device=device, dtype=dtype, requires_grad=True)

    assert circ(state, phi).size() == (2, 2, batch_size)

    state = pyq.zero_state(n_qubits, batch_size=batch_size, device=device, dtype=dtype)

    res = circ(state, phi)
    assert not torch.all(torch.isnan(res))

    # g = torch.autograd.grad(circ, thetas)
    dres_theta = torch.autograd.grad(res, phi, torch.ones_like(res), create_graph=True)[0]
    assert not torch.all(torch.isnan(dres_theta))


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_U_gate(batch_size: int) -> None:
    n_qubits = 1
    u = pyq.U([0], n_qubits)
    x = torch.rand(3, batch_size)
    state = pyq.zero_state(n_qubits, batch_size=batch_size, device="cpu", dtype=torch.cdouble)
    assert not torch.all(torch.isnan(u(state, x)))
