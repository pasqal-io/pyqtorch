from __future__ import annotations

import random

import pytest
import torch

import pyqtorch as pyq
from pyqtorch.primitives import Primitive
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
def test_mutation(op: Primitive) -> None:
    # checking mutation is equivalent to the original forward method
    n_qubits = random.randint(1, 5)
    target = random.randint(0, n_qubits - 1)
    state = random_state(n_qubits)
    gate = op(target)

    primitive_op = Primitive(gate.operation, qubit_support=gate.qubit_support)
    assert torch.allclose(gate(state), primitive_op(state), atol=ATOL)


def test_mutation_swap() -> None:
    # checking mutation is equivalent to the original forward method
    n_qubits = random.randint(2, 5)
    target = random.randint(0, n_qubits - 1)
    target2 = random.choice([i for i in range(n_qubits) if i != target])
    gate = pyq.SWAP(target, target2)
    state = random_state(n_qubits)
    primitive_op = Primitive(gate.operation, qubit_support=gate.qubit_support)
    assert torch.allclose(gate(state), primitive_op(state), atol=ATOL)
