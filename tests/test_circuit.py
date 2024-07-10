from __future__ import annotations

import random

import pytest
import torch
from torch import Tensor

import pyqtorch as pyq
from pyqtorch import run, sample
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.noise import Noise
from pyqtorch.parametric import Parametric
from pyqtorch.primitive import Primitive
from pyqtorch.utils import (
    DensityMatrix,
    product_state,
)


def test_device_inference() -> None:
    ops = [pyq.RX(0), pyq.RX(0)]
    circ = pyq.QuantumCircuit(2, ops)
    nested_circ = pyq.QuantumCircuit(2, [circ, circ])
    assert nested_circ._device is not None


@pytest.mark.parametrize("fn", [pyq.X, pyq.Z, pyq.Y])
def test_scale(fn: pyq.primitive.Primitive) -> None:
    n_qubits = torch.randint(low=1, high=4, size=(1,)).item()
    target = random.choice([i for i in range(n_qubits)])
    state = pyq.random_state(n_qubits)
    gate = fn(target)
    values = {"scale": torch.rand(1)}
    wf = values["scale"] * pyq.QuantumCircuit(2, [gate])(state, {})
    scaledwf_primitive = pyq.Scale(gate, "scale")(state, values)
    scaledwf_composite = pyq.Scale(pyq.Sequence([gate]), "scale")(state, values)
    assert torch.allclose(wf, scaledwf_primitive)
    assert torch.allclose(wf, scaledwf_composite)


def test_add() -> None:
    num_gates = 2
    fns = [pyq.X, pyq.Y, pyq.Z, pyq.S, pyq.H, pyq.T]
    ops = []
    n_qubits = torch.randint(low=1, high=4, size=(1,)).item()
    state = pyq.random_state(n_qubits)
    chosen_gate_ids = torch.randint(0, len(fns) - 1, size=(num_gates,))
    for id in chosen_gate_ids:
        target = random.choice([i for i in range(n_qubits)])
        ops.append(fns[id](target))

    assert torch.allclose(pyq.Add(ops)(state), ops[0](state) + ops[1](state))


def test_merge() -> None:
    ops = [pyq.RX(0, "theta_0"), pyq.RY(0, "theta_1"), pyq.RX(0, "theta_2")]
    circ = pyq.QuantumCircuit(2, ops)
    mergecirc = pyq.Merge(ops)
    state = pyq.random_state(2)
    values = {f"theta_{i}": torch.rand(1) for i in range(3)}
    assert torch.allclose(circ(state, values), mergecirc(state, values))


@pytest.mark.xfail(
    reason="Can only merge single qubit gates acting on the same qubit support."
)
def test_merge_different_sup_expect_fail() -> None:
    ops = [pyq.RX(0, "theta_0"), pyq.RY(1, "theta_1")]
    pyq.Merge(ops)


@pytest.mark.xfail(
    reason="Can only merge single qubit gates acting on the same qubit support."
)
def test_merge_multiqubit_expect_fail() -> None:
    ops = [pyq.CNOT(0, 1), pyq.RY(1, "theta_1")]
    pyq.Merge(ops)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_merge_different_batchsize(batch_size: int) -> None:
    ops = [pyq.X(0), pyq.RX(0, "theta_0")]
    mergecirc = pyq.Merge(ops)
    for bs in [1, batch_size]:
        mergecirc(pyq.random_state(2, bs), {"theta_0": torch.rand(batch_size)})
        mergecirc(pyq.random_state(2, batch_size), {"theta_0": torch.rand(bs)})


def test_merge_nested_dict() -> None:
    ops = [pyq.X(0), pyq.RX(0, "theta_0")]
    mergecirc = pyq.Merge(ops)
    vals = {
        "theta_0": torch.rand(1),
        "theta_2": torch.rand(2),
        "theta_3": torch.rand(2),
    }
    vals["nested"] = vals
    mergecirc(pyq.random_state(2), vals)


@pytest.mark.parametrize("n_qubits", [{"low": 2, "high": 5}], indirect=True)
@pytest.mark.parametrize("batch_size", [{"low": 1, "high": 5}], indirect=True)
def test_noise_circ(
    n_qubits: int,
    batch_size: int,
    random_input_state: Tensor,
    random_gate: Primitive,
    random_noise_gate: Noise,
    random_rotation_gate: Parametric,
) -> None:
    OPERATORS = [random_gate, random_noise_gate, random_rotation_gate]
    random.shuffle(OPERATORS)
    circ = QuantumCircuit(n_qubits, OPERATORS)

    values = {random_rotation_gate.param_name: torch.rand(1)}
    output_state = circ(random_input_state, values)
    assert isinstance(output_state, DensityMatrix)
    assert output_state.shape == torch.Size([2**n_qubits, 2**n_qubits, batch_size])

    diag_sums = []
    for i in range(batch_size):
        diag_batch = torch.diagonal(output_state[:, :, i], dim1=0, dim2=1)
        diag_sums.append(torch.sum(diag_batch))
    diag_sum = torch.stack(diag_sums)
    assert torch.allclose(diag_sum, torch.ones((batch_size,), dtype=torch.cdouble))


def test_sample_run() -> None:
    ops = [pyq.X(0), pyq.X(1)]
    circ = pyq.QuantumCircuit(4, ops)
    wf = run(circ)
    samples = sample(circ)
    assert torch.allclose(wf, product_state("1100"))
    assert torch.allclose(pyq.QuantumCircuit(4, [pyq.I(0)]).run("1100"), wf)
    assert "1100" in samples[0]
