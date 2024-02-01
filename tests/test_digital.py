from __future__ import annotations

import random
from math import log2
from typing import Callable, Tuple

import pytest
import torch

import pyqtorch as pyq
from pyqtorch.apply import apply_operator
from pyqtorch.matrices import IMAT, ZMAT
from pyqtorch.parametric import Parametric
from pyqtorch.utils import product_state

state_000 = product_state("000")
state_001 = product_state("001")
state_100 = product_state("100")
state_101 = product_state("101")
state_110 = product_state("110")
state_111 = product_state("111")
state_0000 = product_state("0000")
state_1110 = product_state("1110")
state_1111 = product_state("1111")


def test_identity() -> None:
    assert torch.allclose(product_state("0"), pyq.I(0)(product_state("0"), None))
    assert torch.allclose(product_state("1"), pyq.I(1)(product_state("1")))


def test_N() -> None:
    null_state = torch.zeros_like(pyq.zero_state(1))
    assert torch.allclose(null_state, pyq.N(0)(product_state("0"), None))
    assert torch.allclose(product_state("1"), pyq.N(0)(product_state("1"), None))


def test_projectors() -> None:
    t0 = torch.tensor([[0.0], [0.0]], dtype=torch.cdouble)
    t1 = torch.tensor([[1.0], [0.0]], dtype=torch.cdouble)
    t2 = torch.tensor([[0.0], [1.0]], dtype=torch.cdouble)
    assert torch.allclose(t1, pyq.Projector(0, ket="0", bra="0")(product_state("0")))
    assert torch.allclose(t0, pyq.Projector(0, ket="0", bra="0")(product_state("1")))
    assert torch.allclose(t2, pyq.Projector(0, ket="1", bra="1")(product_state("1")))
    assert torch.allclose(t0, pyq.Projector(0, ket="1", bra="1")(product_state("0")))
    t00 = torch.tensor([[[1.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]])
    t01 = torch.tensor([[[0.0 + 0.0j], [1.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]])
    t10 = torch.tensor([[[0.0 + 0.0j], [0.0 + 0.0j]], [[1.0 + 0.0j], [0.0 + 0.0j]]])
    t11 = torch.tensor([[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [1.0 + 0.0j]]])
    assert torch.allclose(pyq.Projector((0, 1), ket="00", bra="00")(product_state("00")), t00)
    assert torch.allclose(pyq.Projector((0, 1), ket="10", bra="01")(product_state("01")), t10)
    assert torch.allclose(pyq.Projector((0, 1), ket="01", bra="10")(product_state("10")), t01)
    assert torch.allclose(pyq.Projector((0, 1), ket="11", bra="11")(product_state("11")), t11)
    t000 = torch.tensor(
        [
            [[[1.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        ]
    )
    t100 = torch.tensor(
        [
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
            [[[1.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        ]
    )
    t001 = torch.tensor(
        [
            [[[0.0 + 0.0j], [1.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        ]
    )
    t010 = torch.tensor(
        [
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[1.0 + 0.0j], [0.0 + 0.0j]]],
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
        ]
    )
    t111 = torch.tensor(
        [
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [0.0 + 0.0j]]],
            [[[0.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [1.0 + 0.0j]]],
        ]
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="000", bra="000")(product_state("000")), t000
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="100", bra="001")(product_state("001")), t100
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="010", bra="010")(product_state("010")), t010
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="001", bra="100")(product_state("100")), t001
    )
    assert torch.allclose(
        pyq.Projector((0, 1, 2), ket="111", bra="111")(product_state("111")), t111
    )


def test_CNOT_state00_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CNOT(0, 1)(product_state("00"), None)
    assert torch.equal(product_state("00"), result)


def test_CNOT_state10_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CNOT(0, 1)(product_state("10"), None)
    assert torch.equal(product_state("11"), result)


def test_CNOT_state11_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CNOT(0, 1)(product_state("11"), None)
    assert torch.equal(product_state("10"), result)


def test_CRY_state10_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CRY(0, 1, "theta")(
        product_state("10"), {"theta": torch.tensor([torch.pi])}
    )
    assert torch.allclose(product_state("11"), result)


def test_CRY_state01_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CRY(1, 0, "theta")(
        product_state("01"), {"theta": torch.tensor([torch.pi])}
    )
    assert torch.allclose(product_state("11"), result)


def test_CSWAP_state101_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CSWAP((0, 1), 2)(product_state("101"), None)
    assert torch.allclose(product_state("110"), result)


def test_CSWAP_state110_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CSWAP((0, 1), 2)(product_state("101"), None)
    assert torch.allclose(product_state("110"), result)


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
def test_CSWAP_controlqubits0(initial_state: torch.Tensor, expected_state: torch.Tensor) -> None:
    cswap = pyq.CSWAP((0, 1), 2)
    assert torch.allclose(cswap(initial_state, None), expected_state)


@pytest.mark.parametrize(
    "initial_state,expected_state",
    [
        (state_000, state_000),
        (state_001, state_001),
        (state_100, state_100),
        (state_101, state_101),
        (state_110, state_111),
        (state_1110, state_1111),
    ],
)
def test_Toffoli_controlqubits0(initial_state: torch.Tensor, expected_state: torch.Tensor) -> None:
    n_qubits = int(log2(torch.numel(initial_state)))
    qubits = tuple([i for i in range(n_qubits)])
    toffoli = pyq.Toffoli(qubits[:-1], qubits[-1])
    assert torch.allclose(toffoli(initial_state, None), expected_state)


@pytest.mark.parametrize(
    "initial_state,expects_rotation",
    [
        (state_000, False),
        (state_001, False),
        (state_100, False),
        (state_101, False),
        (state_110, True),
        (state_1110, True),
    ],
)
@pytest.mark.parametrize("gate", ["RX", "RY", "RZ", "PHASE"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_multi_controlled_gates(
    initial_state: torch.Tensor, expects_rotation: bool, batch_size: int, gate: str
) -> None:
    phi = "phi"
    rot_gate = getattr(pyq, gate)
    controlled_rot_gate = getattr(pyq, "C" + gate)
    phi = torch.rand(batch_size)
    n_qubits = int(log2(torch.numel(initial_state)))
    qubits = tuple([i for i in range(n_qubits)])
    op = controlled_rot_gate(qubits[:-1], qubits[-1], "phi")
    out = op(initial_state, {"phi": phi})
    expected_state = (
        rot_gate(qubits[-1], "phi")(initial_state, {"phi": phi})
        if expects_rotation
        else initial_state
    )
    assert torch.allclose(out, expected_state)


@pytest.mark.parametrize("state_fn", [pyq.random_state, pyq.zero_state, pyq.uniform_state])
def test_parametric_phase_hamevo(
    state_fn: Callable, batch_size: int = 1, n_qubits: int = 1
) -> None:
    target = 0
    state = state_fn(n_qubits, batch_size=batch_size)
    phi = torch.rand(1, dtype=torch.cdouble)
    H = (ZMAT - IMAT) / 2
    hamevo = pyq.HamiltonianEvolution(qubit_support=(target,), n_qubits=n_qubits)
    phase = pyq.PHASE(target, "phi")
    assert torch.allclose(phase(state, {"phi": phi}), hamevo(H, phi, state))


@pytest.mark.parametrize("state_fn", [pyq.random_state, pyq.zero_state, pyq.uniform_state])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_qubits", [1, 2, 4])
def test_parametrized_phase_gate(state_fn: Callable, batch_size: int, n_qubits: int) -> None:
    target: int = torch.randint(low=0, high=n_qubits, size=(1,)).item()
    state = state_fn(n_qubits, batch_size=batch_size)
    phi = torch.tensor([torch.pi / 2], dtype=torch.cdouble)
    phase = pyq.PHASE(target, "phi")
    constant_phase = pyq.S(target)
    assert torch.allclose(phase(state, {"phi": phi}), constant_phase(state, None))


def test_dagger_single_qubit() -> None:
    for cls in [pyq.X, pyq.Y, pyq.Z, pyq.S, pyq.H, pyq.T, pyq.RX, pyq.RY, pyq.RZ, pyq.PHASE]:
        n_qubits = torch.randint(low=1, high=4, size=(1,)).item()
        target = random.choice([i for i in range(n_qubits)])
        state = pyq.random_state(n_qubits)
        for param_name in ["theta", ""]:
            if issubclass(cls, Parametric):
                op = cls(target, param_name)  # type: ignore[arg-type]
            else:
                op = cls(target)  # type: ignore[assignment, call-arg]
            values = {param_name: torch.rand(1)} if param_name == "theta" else torch.rand(1)
            new_state = apply_operator(state, op.unitary(values), [target])
            daggered_back = apply_operator(new_state, op.dagger(values), [target])
            assert torch.allclose(daggered_back, state)


def test_dagger_nqubit() -> None:
    for cls in [pyq.SWAP, pyq.CNOT, pyq.CY, pyq.CZ, pyq.CRX, pyq.CRY, pyq.CRZ, pyq.CPHASE]:
        qubit_support: Tuple[int, ...]
        n_qubits = torch.randint(low=3, high=8, size=(1,)).item()
        target = random.choice([i for i in range(n_qubits - 2)])
        state = pyq.random_state(n_qubits)
        for param_name in ["theta", ""]:
            if isinstance(cls, (pyq.CSWAP, pyq.Toffoli)):
                op = cls((target - 2, target - 1), target)
                qubit_support = (target + 2, target + 1, target)
            elif issubclass(cls, Parametric):
                op = cls(target - 1, target, param_name)  # type: ignore[arg-type]
                qubit_support = (target + 1, target)
            else:
                op = cls(target - 1, target)  # type: ignore[call-arg]
                qubit_support = (target + 1, target)
            values = {param_name: torch.rand(1)} if param_name == "theta" else torch.rand(1)
            new_state = apply_operator(state, op.unitary(values), qubit_support)
            daggered_back = apply_operator(new_state, op.dagger(values), qubit_support)
            assert torch.allclose(daggered_back, state)


def test_U() -> None:
    n_qubits = torch.randint(low=1, high=8, size=(1,)).item()
    target = random.choice([i for i in range(n_qubits)])
    params = ["phi", "theta", "omega"]
    u = pyq.U(target, *params)
    values = {param: torch.rand(1) for param in params}
    state = pyq.random_state(n_qubits)
    assert torch.allclose(
        u(state, values), pyq.QuantumCircuit(n_qubits, u.digital_decomposition())(state, values)
    )
