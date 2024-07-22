from __future__ import annotations

import pytest
import torch

import pyqtorch as pyq
from pyqtorch.utils import ATOL, DropoutMode


@pytest.mark.parametrize(
    "dropout_mode",
    [DropoutMode.ROTATIONAL, DropoutMode.ENTANGLING, DropoutMode.CANONICAL_FWD],
)
def test_dropout(
    dropout_mode: DropoutMode, n_qubits: int = 4, theta: float = 0.6
) -> None:
    """Test when dropout prob = 1.0 if all gates the dropout method affects are dropped."""
    circ_ops_1 = [pyq.RY(i, f"theta_{str(i)}") for i in range(n_qubits)]
    circ_ops_2 = [pyq.CNOT(i, i + 1) for i in range(n_qubits - 1)]

    drop_circ_ops = circ_ops_1 + circ_ops_2

    operations = {
        DropoutMode.ROTATIONAL: circ_ops_2,
        DropoutMode.ENTANGLING: circ_ops_1,
        DropoutMode.CANONICAL_FWD: [],
    }

    values = {
        f"theta_{str(i)}": torch.tensor([theta], requires_grad=True)
        for i in range(n_qubits)
    }

    state = pyq.zero_state(n_qubits=n_qubits)
    circ = pyq.QuantumCircuit(n_qubits=n_qubits, operations=operations[dropout_mode])
    dropout_circ = pyq.DropoutQuantumCircuit(
        n_qubits=n_qubits,
        operations=drop_circ_ops,
        dropout_mode=dropout_mode,
        dropout_prob=1.0,
    )
    obs = pyq.Observable([pyq.Z(1)])

    exp_circ = pyq.expectation(circuit=circ, state=state, values=values, observable=obs)
    exp_dropout_circ = pyq.expectation(
        circuit=dropout_circ, state=state, values=values, observable=obs
    )

    assert torch.allclose(exp_circ, exp_dropout_circ, atol=ATOL)
