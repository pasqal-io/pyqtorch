from __future__ import annotations

import random

import pytest
import torch

import pyqtorch as pyq
from pyqtorch.primitives import ControlledPrimitive, Primitive
from pyqtorch.utils import (
    ATOL,
    random_state,
)


@pytest.mark.parametrize(
    "op",
    [
        pyq.X,
        pyq.Z,
        pyq.T,
        pyq.S,
        pyq.SDagger,
        pyq.Y,
        pyq.N,
    ],
)
def test_mutation_single(op: Primitive) -> None:
    # checking mutation is equivalent to the original forward method
    n_qubits = random.randint(1, 5)
    target = random.randint(0, n_qubits - 1)
    state = random_state(n_qubits)
    gate = op(target)

    primitive_op = Primitive(gate.operation, qubit_support=gate.target)
    assert torch.allclose(gate(state), primitive_op(state), atol=ATOL)


@pytest.mark.parametrize(
    "op, op_str",
    [
        (pyq.CNOT, "X"),
        (pyq.CZ, "Z"),
        (pyq.Toffoli, "X"),
    ],
)
def test_mutation_controlled(op: Primitive, op_str: str) -> None:
    # checking mutation is equivalent to the original forward method
    n_qubits = random.randint(3, 6)
    target = random.randint(0, n_qubits - 1)
    control = random.choice([i for i in range(n_qubits) if i != target])
    if op == pyq.Toffoli:
        control = (  # type: ignore[assignment]
            control,
            random.choice([i for i in range(n_qubits) if i not in (target, control)]),
        )
    state = random_state(n_qubits)
    gate = op(control, target)

    primitive_op = ControlledPrimitive(op_str, control=gate.control, target=gate.target)
    assert torch.allclose(gate(state), primitive_op(state), atol=ATOL)


def test_mutation_swap() -> None:
    # checking mutation is equivalent to the original forward method
    n_qubits = random.randint(2, 5)
    target = random.randint(0, n_qubits - 1)
    target2 = random.choice([i for i in range(n_qubits) if i != target])
    gate = pyq.SWAP(target, target2)
    state = random_state(n_qubits)
    primitive_op = Primitive(gate.operation, qubit_support=gate.target)
    assert torch.allclose(gate(state), primitive_op(state), atol=ATOL)
