from __future__ import annotations

from typing import Callable

import pytest
import torch
from torch import Tensor

import pyqtorch.core as func_pyq
import pyqtorch.modules as pyq
from pyqtorch.core.utils import OPERATIONS_DICT
from pyqtorch.modules.abstract import AbstractGate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.cdouble

state_000 = pyq.zero_state(3, device=DEVICE, dtype=DTYPE)
state_001 = pyq.X(qubits=[2], n_qubits=3)(state_000)
state_100 = pyq.X(qubits=[0], n_qubits=3)(state_000)
state_101 = pyq.X(qubits=[2], n_qubits=3)(pyq.X(qubits=[0], n_qubits=3)(state_000))
state_110 = pyq.X(qubits=[1], n_qubits=3)(pyq.X(qubits=[0], n_qubits=3)(state_000))
state_111 = pyq.X(qubits=[2], n_qubits=3)(
    pyq.X(qubits=[1], n_qubits=3)(pyq.X(qubits=[0], n_qubits=3)(state_000))
)

state_0000 = pyq.zero_state(4, device=DEVICE, dtype=DTYPE)
state_1110 = pyq.X(qubits=[0], n_qubits=4)(
    pyq.X(qubits=[1], n_qubits=4)(pyq.X(qubits=[2], n_qubits=4)(state_0000))
)
state_1111 = pyq.X(qubits=[0], n_qubits=4)(
    pyq.X(qubits=[1], n_qubits=4)(
        pyq.X(qubits=[2], n_qubits=4)(pyq.X(qubits=[3], n_qubits=4)(state_0000))
    )
)


@pytest.mark.parametrize("batch_size", [i for i in range(1, 2, 10)])
@pytest.mark.parametrize("n_qubits", [i for i in range(1, 6)])
@pytest.mark.parametrize("gate", ["X", "Y", "Z", "H", "I", "S", "T", "Sdagger"])
def test_constant_gates(batch_size: int, n_qubits: int, gate: str) -> None:
    dtype = torch.cdouble
    qubits = [torch.randint(low=0, high=n_qubits, size=(1,)).item()]

    state = pyq.random_state(n_qubits, batch_size=batch_size, device=DEVICE, dtype=DTYPE)
    Op = getattr(pyq, gate)
    FuncOp = getattr(func_pyq.operation, gate)

    func_out = FuncOp(state, qubits, n_qubits)
    op = Op(qubits, n_qubits).to(device=DEVICE, dtype=dtype)
    mod_out = op(state)

    assert torch.allclose(func_out, mod_out)


@pytest.mark.parametrize("batch_size", [i for i in range(1, 2, 10)])
@pytest.mark.parametrize("n_qubits", [i for i in range(1, 6)])
@pytest.mark.parametrize("gate", ["RX", "RY", "RZ"])
def test_parametrized_gates(batch_size: int, n_qubits: int, gate: str) -> None:
    qubits = [torch.randint(low=0, high=n_qubits, size=(1,)).item()]

    state = pyq.random_state(n_qubits, batch_size=batch_size, device=DEVICE, dtype=DTYPE)
    phi = torch.rand(batch_size, device=DEVICE, dtype=DTYPE)

    Op = getattr(pyq, gate)
    FuncOp = getattr(func_pyq.batched_operation, f"batched{gate}")

    func_out = FuncOp(phi, state, qubits, n_qubits)
    op = Op(qubits, n_qubits).to(device=DEVICE, dtype=DTYPE)
    mod_out = op(state, phi)

    assert torch.allclose(func_out, mod_out)


@pytest.mark.parametrize("batch_size", [i for i in range(1, 2, 10)])
@pytest.mark.parametrize("n_qubits", [i for i in range(2, 6)])
@pytest.mark.parametrize("gate", ["CRX", "CRY", "CRZ"])
def test_controlled_parametrized_gates(batch_size: int, n_qubits: int, gate: str) -> None:
    qubits = torch.randint(low=0, high=n_qubits, size=(2,))

    while qubits[0] == qubits[1]:
        qubits[1] = torch.randint(low=0, high=n_qubits, size=(1,))

    qubits = [qubits[0].item(), qubits[1].item()]

    state = pyq.random_state(n_qubits, batch_size=batch_size, device=DEVICE, dtype=DTYPE)
    phi = torch.rand(batch_size, device=DEVICE, dtype=DTYPE)

    Op = getattr(pyq, gate)
    BatchedOP = getattr(func_pyq.batched_operation, f"batched{gate}")

    func_out = BatchedOP(phi, state, qubits, n_qubits)
    op = Op(qubits, n_qubits).to(device=DEVICE, dtype=DTYPE)
    mod_out = op(state, phi)

    assert torch.allclose(func_out, mod_out)


@pytest.mark.parametrize("batch_size", [i for i in range(1, 2, 10)])
@pytest.mark.parametrize("n_qubits", [i for i in range(2, 6)])
def test_circuit(batch_size: int, n_qubits: int) -> None:
    ops = [
        pyq.X([0], n_qubits),
        pyq.X([1], n_qubits),
        pyq.RX([1], n_qubits),
        pyq.CNOT([0, 1], n_qubits),
    ]
    circ = pyq.QuantumCircuit(n_qubits, ops).to(device=DEVICE, dtype=DTYPE)

    state = pyq.random_state(n_qubits, batch_size)
    phi = torch.rand(batch_size, device=DEVICE, dtype=DTYPE, requires_grad=True)

    assert circ(state, phi).size() == tuple(2 for _ in range(n_qubits)) + (batch_size,)

    state = pyq.random_state(n_qubits, batch_size=batch_size, device=DEVICE, dtype=DTYPE)

    res = circ(state, phi)
    assert not torch.all(torch.isnan(res))
    dres_theta = torch.autograd.grad(res, phi, torch.ones_like(res), create_graph=True)[0]
    assert not torch.all(torch.isnan(dres_theta))


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_empty_circuit(batch_size: int) -> None:
    n_qubits = 2
    ops: list = []
    circ = pyq.QuantumCircuit(n_qubits, ops).to(device=DEVICE, dtype=DTYPE)

    state = circ.init_state(batch_size)
    phi = torch.rand(batch_size, device=DEVICE, dtype=DTYPE, requires_grad=True)

    assert circ(state, phi).size() == (2, 2, batch_size)

    state = pyq.random_state(n_qubits, batch_size=batch_size, device=DEVICE, dtype=DTYPE)

    res = circ(state, phi)
    assert not torch.all(torch.isnan(res))


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_U_gate(batch_size: int) -> None:
    n_qubits = 1
    u = pyq.U([0], n_qubits)
    x = torch.rand(3, batch_size)
    state = pyq.random_state(n_qubits, batch_size=batch_size, device=DEVICE, dtype=DTYPE)
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


@pytest.mark.parametrize(
    "initial_state,expected_state",
    [
        (state_000, state_000),
        (state_001, state_001),
        (state_100, state_100),
        (state_101, state_110),
        (state_110, state_101),
    ],
)
def test_CSWAP_controlqubits0(initial_state: Tensor, expected_state: Tensor) -> None:
    print(initial_state.shape)
    print(expected_state.shape)
    n_qubits = 3
    cswap = pyq.CSWAP([0, 1, 2], n_qubits)
    assert torch.allclose(cswap(initial_state), expected_state)


@pytest.mark.parametrize(
    "initial_state,expected_state",
    [
        (state_000, state_000),
        (state_001, state_001),
        (state_100, state_100),
        (state_101, state_101),
        (state_110, state_111),
    ],
)
def test_Toffoli_controlqubits0(initial_state: Tensor, expected_state: Tensor) -> None:
    n_qubits = 3
    toffoli = pyq.Toffoli([0, 1, 2], n_qubits)
    assert torch.allclose(toffoli(initial_state), expected_state)


def test_4qubit_Toffoli() -> None:
    n_qubits = 4
    toffoli = pyq.Toffoli([0, 1, 2, 3], n_qubits)
    assert torch.allclose(toffoli(state_1110), state_1111)


@pytest.mark.parametrize("state_fn", [pyq.random_state, pyq.zero_state, pyq.uniform_state])
@pytest.mark.parametrize("n_qubits", [i for i in range(1, 8)])
@pytest.mark.parametrize("batch_size", [i for i in range(1, 8)])
def test_isnormalized_states(state_fn: Callable, n_qubits: int, batch_size: int) -> None:
    state = state_fn(n_qubits, batch_size, device=DEVICE, dtype=DTYPE)
    assert pyq.is_normalized(state)


@pytest.mark.parametrize("n_qubits", [i for i in range(1, 8)])
@pytest.mark.parametrize("batch_size", [i for i in range(1, 8)])
def test_state_shapes(n_qubits: int, batch_size: int) -> None:
    zero = pyq.zero_state(n_qubits, batch_size, device=DEVICE, dtype=DTYPE)
    uni = pyq.uniform_state(n_qubits, batch_size, device=DEVICE, dtype=DTYPE)
    rand = pyq.random_state(n_qubits, batch_size, device=DEVICE, dtype=DTYPE)
    assert zero.shape == rand.shape and uni.shape == rand.shape


@pytest.mark.parametrize("state_fn", [pyq.random_state, pyq.zero_state, pyq.uniform_state])
@pytest.mark.parametrize("n_qubits", [i for i in range(1, 8)])
@pytest.mark.parametrize("batch_size", [i for i in range(1, 8)])
def test_overlap_states_batch_nqubits(state_fn: Callable, n_qubits: int, batch_size: int) -> None:
    state = state_fn(n_qubits, batch_size, device=DEVICE, dtype=DTYPE)
    assert torch.allclose(
        pyq.overlap(state, state),
        torch.ones(batch_size),
    )


@pytest.mark.parametrize("state_fn", [pyq.random_state, pyq.zero_state, pyq.uniform_state])
@pytest.mark.parametrize("batch_size", [i for i in range(1, 2, 10)])
@pytest.mark.parametrize("n_qubits", [i for i in range(1, 6)])
def test_parametrized_phase_gate(state_fn: Callable, batch_size: int, n_qubits: int) -> None:
    qubits = [torch.randint(low=0, high=n_qubits, size=(1,)).item()]
    state = state_fn(n_qubits, batch_size=batch_size, device=DEVICE, dtype=DTYPE)
    phi = torch.tensor([torch.pi / 2], dtype=torch.cdouble)
    phase = pyq.PHASE(qubits, n_qubits).to(device=DEVICE, dtype=DTYPE)
    constant_phase = pyq.S(qubits, n_qubits).to(device=DEVICE, dtype=DTYPE)
    assert torch.allclose(phase(state, phi), constant_phase(state, phi))


@pytest.mark.parametrize("state_fn", [pyq.random_state, pyq.zero_state, pyq.uniform_state])
def test_parametric_phase_hamevo(
    state_fn: Callable, batch_size: int = 1, n_qubits: int = 1
) -> None:
    qubits = [0]
    state = state_fn(n_qubits, batch_size=batch_size, device=DEVICE, dtype=DTYPE)
    phi = torch.rand(1, dtype=torch.cdouble)
    H = (OPERATIONS_DICT["Z"] - OPERATIONS_DICT["I"]) / 2
    hamevo = pyq.HamEvoExp(H, phi, qubits=qubits, n_qubits=n_qubits)
    phase = pyq.PHASE(qubits, n_qubits).to(device=DEVICE, dtype=DTYPE)
    assert torch.allclose(phase(state, phi), hamevo(state))
