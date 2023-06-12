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
def test_empy_circuit(batch_size: int) -> None:
    n_qubits = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.cdouble

    ops: list = []
    circ = pyq.QuantumCircuit(n_qubits, ops).to(device=device, dtype=dtype)

    state = circ.init_state(batch_size)
    phi = torch.rand(batch_size, device=device, dtype=dtype, requires_grad=True)

    assert circ(state, phi).size() == (2, 2, batch_size)

    state = pyq.zero_state(n_qubits, batch_size=batch_size, device=device, dtype=dtype)

    res = circ(state, phi)
    assert not torch.all(torch.isnan(res))


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_U_gate(batch_size: int) -> None:
    n_qubits = 1
    u = pyq.U([0], n_qubits)
    x = torch.rand(3, batch_size)
    state = pyq.zero_state(n_qubits, batch_size=batch_size, device="cpu", dtype=torch.cdouble)
    assert not torch.all(torch.isnan(u(state, x)))


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_gate_composition(n_qubits: int) -> None:
    X = pyq.X(torch.randint(0, n_qubits, (1,)).tolist(), n_qubits)
    Y = pyq.Y(torch.randint(0, n_qubits, (1,)).tolist(), n_qubits)
    XYcirc = X * Y
    XYcirc_ref = pyq.QuantumCircuit(n_qubits, [X, Y])
    assert XYcirc.is_same_circuit(XYcirc_ref)

    RX = pyq.RX(torch.randint(0, n_qubits, (1,)).tolist(), n_qubits)
    RY = pyq.RY(torch.randint(0, n_qubits, (1,)).tolist(), n_qubits)
    RXRYcirc = RX * RY
    RXRYcirc_ref = pyq.QuantumCircuit(n_qubits, [RX, RY])
    assert RXRYcirc.is_same_circuit(RXRYcirc_ref)

    r1, r2, r3 = 1, n_qubits, max(n_qubits - 1, 1)
    Z1 = pyq.Z(torch.randint(0, r1, (1,)).tolist(), r1)
    Y = pyq.Y(torch.randint(0, r2, (1,)).tolist(), r2)
    Z2 = pyq.Z(torch.randint(0, r3, (1,)).tolist(), r3)

    ZYZcirc = Z1 * Y * Z2
    ZYZcirc_ref = pyq.QuantumCircuit(max(r1, r2, r3), [Z1, Y, Z2])
    assert ZYZcirc.is_same_circuit(ZYZcirc_ref)

    rr1, rr2, rr3 = 1, n_qubits, max(n_qubits - 1, 1)
    RZ1 = pyq.RZ(torch.randint(0, rr1, (1,)).tolist(), rr1)
    RY1 = pyq.RY(torch.randint(0, rr2, (1,)).tolist(), rr2)
    RZ2 = pyq.RZ(torch.randint(0, rr3, (1,)).tolist(), rr3)

    Ucirc = RZ1 * RY1 * RZ2
    Ucirc_ref = pyq.QuantumCircuit(max(r1, r2, r3), [RZ1, RY1, RZ2])
    assert Ucirc.is_same_circuit(Ucirc_ref)


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_QuantumCircuit_composition(n_qubits: int) -> None:
    r = torch.randint(1, n_qubits + 1, (1,)).tolist()
    RX = pyq.RX(r, n_qubits)

    RXcirc = RX * pyq.QuantumCircuit(1, [pyq.X([0], 1), pyq.Y([0], 1)])
    RXcirc_ref = pyq.QuantumCircuit(n_qubits, [RX, pyq.X([0], 1), pyq.Y([0], 1)])
    assert RXcirc.is_same_circuit(RXcirc_ref)

    circRX = pyq.QuantumCircuit(1, [pyq.X([0], 1), pyq.Y([0], 1)]) * RX
    circRX_ref = pyq.QuantumCircuit(n_qubits, [pyq.X([0], 1), pyq.Y([0], 1), RX])
    assert circRX.is_same_circuit(circRX_ref)

    circ_mult = pyq.QuantumCircuit(1, [pyq.X([0], 1), pyq.Y([0], 1)]) * pyq.QuantumCircuit(
        n_qubits, [RX, pyq.RY([0], 1)]
    )
    circ_mult_ref = pyq.QuantumCircuit(n_qubits, [pyq.X([0], 1), pyq.Y([0], 1), RX, pyq.RY([0], 1)])
    assert circ_mult.is_same_circuit(circ_mult_ref)
