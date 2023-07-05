from __future__ import annotations

import pytest
import torch

import pyqtorch.core as func_pyq
import pyqtorch.modules as pyq
from pyqtorch.modules.abstract import AbstractGate


@pytest.mark.parametrize("gate", ["X", "Y", "Z", "H", "I", "S", "T", "Sdagger"])
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
def test_empty_circuit(batch_size: int) -> None:
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


@pytest.mark.parametrize(
    "a,b,val",
    [
        (pyq.X([0], 1), pyq.X([0], 1), True),
        (pyq.RY([0], 1), pyq.RY([0], 1), True),
        (pyq.RY([0], 1), pyq.RY([0], 1), True),
        (pyq.X([0], 1), pyq.Y([0], 1), False),
        (pyq.RX([0], 2), pyq.RX([0], 1), False),
        (pyq.RX([1], 1), pyq.RX([0], 1), False),
    ],
)
def test_gate_equality(a: AbstractGate, b: AbstractGate, val: bool) -> None:
    x = a == b
    assert x == val


def test_circuit_equality() -> None:
    c1 = pyq.QuantumCircuit(2, [pyq.RX([0], 2), pyq.Z([1], 2)])
    c2 = pyq.QuantumCircuit(2, [pyq.RX([0], 2), pyq.Z([1], 2)])
    assert c1 == c2

    c3 = pyq.QuantumCircuit(2, [pyq.RX([1], 2), pyq.Z([1], 2)])
    assert c1 != c3

    c4 = pyq.QuantumCircuit(2, [pyq.Z([1], 2)])
    assert c3 != c4

    c5 = pyq.QuantumCircuit(2, [pyq.RX([0], 2)])
    assert c1 != c5


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_gate_composition(n_qubits: int) -> None:
    x = pyq.X(torch.randint(0, n_qubits, (1,)).tolist(), n_qubits)
    y = pyq.Y(torch.randint(0, n_qubits, (1,)).tolist(), n_qubits)
    circ = x * y
    truth = pyq.QuantumCircuit(n_qubits, [x, y])
    assert circ == truth

    rx = pyq.RX(torch.randint(0, n_qubits, (1,)).tolist(), n_qubits)
    ry = pyq.RY(torch.randint(0, n_qubits, (1,)).tolist(), n_qubits)
    circ = rx * ry
    truth = pyq.QuantumCircuit(n_qubits, [rx, ry])
    assert circ == truth

    r1, r2, r3 = 1, n_qubits, max(n_qubits - 1, 1)
    z1 = pyq.Z(torch.randint(0, r1, (1,)).tolist(), r1)
    y = pyq.Y(torch.randint(0, r2, (1,)).tolist(), r2)
    z2 = pyq.Z(torch.randint(0, r3, (1,)).tolist(), r3)
    circ = z1 * y * z2
    truth = pyq.QuantumCircuit(max(r1, r2, r3), [z1, y, z2])
    assert circ == truth

    rr1, rr2, rr3 = 1, n_qubits, max(n_qubits - 1, 1)
    rz1 = pyq.RZ(torch.randint(0, rr1, (1,)).tolist(), rr1)
    ry1 = pyq.RY(torch.randint(0, rr2, (1,)).tolist(), rr2)
    rz2 = pyq.RZ(torch.randint(0, rr3, (1,)).tolist(), rr3)

    circ = rz1 * ry1 * rz2
    truth = pyq.QuantumCircuit(max(r1, r2, r3), [rz1, ry1, rz2])
    assert circ == truth


@pytest.mark.parametrize("n_qubits", [1, 2, 3])
def test_circuit_composition(n_qubits: int) -> None:
    r = torch.randint(1, n_qubits + 1, (1,)).tolist()
    rx = pyq.RX(r, n_qubits)

    circ = rx * pyq.QuantumCircuit(1, [pyq.X([0], 1), pyq.Y([0], 1)])
    truth = pyq.QuantumCircuit(n_qubits, [rx, pyq.X([0], 1), pyq.Y([0], 1)])
    assert circ == truth

    circ = pyq.QuantumCircuit(1, [pyq.X([0], 1), pyq.Y([0], 1)]) * rx
    truth = pyq.QuantumCircuit(n_qubits, [pyq.X([0], 1), pyq.Y([0], 1), rx])
    assert circ == truth

    circ = pyq.QuantumCircuit(1, [pyq.X([0], 1), pyq.Y([0], 1)]) * pyq.QuantumCircuit(
        n_qubits, [rx, pyq.RY([0], 1)]
    )
    truth = pyq.QuantumCircuit(n_qubits, [pyq.X([0], 1), pyq.Y([0], 1), rx, pyq.RY([0], 1)])
    assert circ == truth
