from __future__ import annotations

import torch

import pyqtorch as pyq
from pyqtorch.utils import DropoutMode


def test_rotational_dropout() -> None:
    n_qubits = 4
    circ_ops = [pyq.CNOT(0, 1), pyq.CNOT(1, 2), pyq.CNOT(2, 3)]
    drop_circ_ops = [
        pyq.RY(0, "theta_0"),
        pyq.RY(1, "theta_1"),
        pyq.RY(2, "theta_2"),
        pyq.RY(3, "theta_3"),
        pyq.CNOT(0, 1),
        pyq.CNOT(1, 2),
        pyq.CNOT(2, 3),
    ]

    theta_0_value = 0.6
    values = {
        "theta_0": torch.tensor([theta_0_value], requires_grad=True),
        "theta_1": torch.tensor([theta_0_value], requires_grad=True),
        "theta_2": torch.tensor([theta_0_value], requires_grad=True),
        "theta_3": torch.tensor([theta_0_value], requires_grad=True),
    }

    state = pyq.zero_state(n_qubits=n_qubits)
    circ = pyq.QuantumCircuit(n_qubits=n_qubits, operations=circ_ops)
    dropout_circ = pyq.DropoutQuantumCircuit(
        n_qubits=n_qubits,
        operations=drop_circ_ops,
        dropout_mode=DropoutMode.ROTATIONAL,
        dropout_prob=1.0,
    )
    obs = pyq.QuantumCircuit(n_qubits=n_qubits, operations=[pyq.Z(1)])

    exp_circ = pyq.expectation(circuit=circ, state=state, values=values, observable=obs)
    exp_dropout_circ = pyq.expectation(
        circuit=dropout_circ, state=state, values=values, observable=obs
    )

    assert exp_circ == exp_dropout_circ


def test_entangling_dropout() -> None:
    n_qubits = 4
    circ_ops = [
        pyq.RY(0, "theta_0"),
        pyq.RY(1, "theta_1"),
        pyq.RY(2, "theta_2"),
        pyq.RY(3, "theta_3"),
    ]
    drop_circ_ops = [
        pyq.RY(0, "theta_0"),
        pyq.RY(1, "theta_1"),
        pyq.RY(2, "theta_2"),
        pyq.RY(3, "theta_3"),
        pyq.CNOT(0, 1),
        pyq.CNOT(1, 2),
        pyq.CNOT(2, 3),
    ]

    theta_value = 0.6
    values = {
        "theta_0": torch.tensor([theta_value], requires_grad=True),
        "theta_1": torch.tensor([theta_value], requires_grad=True),
        "theta_2": torch.tensor([theta_value], requires_grad=True),
        "theta_3": torch.tensor([theta_value], requires_grad=True),
    }

    state = pyq.zero_state(n_qubits=n_qubits)
    circ = pyq.QuantumCircuit(n_qubits=n_qubits, operations=circ_ops)
    dropout_circ = pyq.DropoutQuantumCircuit(
        n_qubits=n_qubits,
        operations=drop_circ_ops,
        dropout_mode=DropoutMode.ENTANGLING,
        dropout_prob=1.0,
    )
    obs = pyq.QuantumCircuit(n_qubits=n_qubits, operations=[pyq.Z(1)])

    exp_circ = pyq.expectation(circuit=circ, state=state, values=values, observable=obs)
    exp_dropout_circ = pyq.expectation(
        circuit=dropout_circ, state=state, values=values, observable=obs
    )

    assert exp_circ == exp_dropout_circ


def test_canonical_fwd_dropout() -> None:
    n_qubits = 4
    drop_circ_ops = [
        pyq.RY(0, "theta_0"),
        pyq.RY(1, "theta_1"),
        pyq.RY(2, "theta_2"),
        pyq.RY(3, "theta_3"),
        pyq.CNOT(0, 1),
        pyq.CNOT(1, 2),
        pyq.CNOT(2, 3),
    ]

    theta_0_value = 0.6
    values = {
        "theta_0": torch.tensor([theta_0_value], requires_grad=True),
        "theta_1": torch.tensor([theta_0_value], requires_grad=True),
        "theta_2": torch.tensor([theta_0_value], requires_grad=True),
        "theta_3": torch.tensor([theta_0_value], requires_grad=True),
    }

    state = pyq.random_state(n_qubits=n_qubits)
    circ = pyq.QuantumCircuit(n_qubits=n_qubits, operations=[])
    dropout_circ = pyq.DropoutQuantumCircuit(
        n_qubits=n_qubits,
        operations=drop_circ_ops,
        dropout_mode=DropoutMode.CANONICAL_FWD,
        dropout_prob=1.0,
    )
    obs = pyq.QuantumCircuit(n_qubits=n_qubits, operations=[pyq.Z(1)])

    exp_circ = pyq.expectation(circuit=circ, state=state, values=values, observable=obs)
    exp_dropout_circ = pyq.expectation(
        circuit=dropout_circ, state=state, values=values, observable=obs
    )

    assert exp_circ == exp_dropout_circ
