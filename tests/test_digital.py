from __future__ import annotations

from math import log2
from typing import Callable

import pytest
import torch

import pyqtorch as pyq
from pyqtorch.matrices import IMAT, ZMAT
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


@pytest.mark.xfail
def test_N() -> None:
    assert torch.allclose(product_state("0"), pyq.N(0)(product_state("0"), None))
    assert torch.allclose(product_state("0"), pyq.N(0)(product_state("1"), None))


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
    result: torch.Tensor = pyq.CSWAP([0, 1], 2)(product_state("101"), None)
    assert torch.allclose(product_state("110"), result)


def test_CSWAP_state110_controlqubit_0() -> None:
    result: torch.Tensor = pyq.CSWAP([0, 1], 2)(product_state("101"), None)
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
    cswap = pyq.CSWAP(
        [
            0,
            1,
        ],
        2,
    )
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
    qubits = [i for i in range(n_qubits)]
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
    qubits = [i for i in range(n_qubits)]
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
    hamevo = pyq.HamiltonianEvolution(H, phi, qubit_support=[target], n_qubits=n_qubits)
    phase = pyq.PHASE(target, "phi")
    assert torch.allclose(phase(state, {"phi": phi}), hamevo(state))


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
