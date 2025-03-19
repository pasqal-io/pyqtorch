from __future__ import annotations

import random

import pytest
import torch

import pyqtorch as pyq
from pyqtorch import run, sample
from pyqtorch.noise import DigitalNoiseProtocol, DigitalNoiseType
from pyqtorch.utils import (
    DensityMatrix,
    density_mat,
    product_state,
    todense_tensor,
)


def test_device_inference() -> None:
    ops = [pyq.RX(0), pyq.RX(0)]
    circ = pyq.QuantumCircuit(2, ops)
    nested_circ = pyq.QuantumCircuit(2, [circ, circ])
    assert nested_circ._device is not None


@pytest.mark.parametrize("fn", [pyq.X, pyq.Z, pyq.Y])
def test_scale(fn: pyq.primitives.Primitive) -> None:
    n_qubits = torch.randint(low=1, high=4, size=(1,)).item()
    target = random.choice([i for i in range(n_qubits)])
    state = pyq.random_state(n_qubits)
    gate = fn(target)
    values = {"scale": torch.rand(1)}
    wf = values["scale"] * pyq.QuantumCircuit(2, [gate])(state, {})

    scale_primitive = pyq.Scale(gate, "scale")
    scale_composite = pyq.Scale(pyq.Sequence([gate]), "scale")

    if gate.is_diagonal:
        assert scale_composite.is_diagonal
        assert scale_primitive.is_diagonal

    scaledwf_primitive = scale_primitive(state, values)
    scaledwf_composite = scale_composite(state, values)
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

    # test diagonal ops
    diagonal_add = pyq.Add(
        [pyq.Z(random.choice([i for i in range(n_qubits)])) for _ in range(num_gates)]
    )
    assert diagonal_add.is_diagonal
    tensor_add = diagonal_add.tensor()
    tensor_add_diagonal = diagonal_add.tensor(diagonal=True)
    dense_diagonal = todense_tensor(tensor_add_diagonal)
    assert torch.allclose(tensor_add, dense_diagonal)


def test_merge() -> None:
    ops = [pyq.RX(0, "theta_0"), pyq.RY(0, "theta_1"), pyq.RX(0, "theta_2")]
    circ = pyq.QuantumCircuit(2, ops)
    mergecirc = pyq.Merge(ops)
    state = pyq.random_state(2)
    values = {f"theta_{i}": torch.rand(1) for i in range(3)}
    assert torch.allclose(circ(state, values), mergecirc(state, values))

    # test with density matrices
    state = density_mat(state)
    circ_out = circ(state, values)
    mergecirc_out = mergecirc(state, values)
    assert isinstance(circ_out, DensityMatrix)
    assert isinstance(mergecirc_out, DensityMatrix)
    assert torch.allclose(circ_out, mergecirc_out)

    # test diagonal merge
    ops = [pyq.RZ(0, "theta_0"), pyq.RZ(0, "theta_1")]
    circ = pyq.QuantumCircuit(2, ops)
    mergecirc = pyq.Merge(ops)
    assert mergecirc.is_diagonal
    tensor_merge = mergecirc.tensor(values)
    tensor_merge_diagonal = mergecirc.tensor(values, diagonal=True)
    dense_diagonal = todense_tensor(tensor_merge_diagonal)
    assert torch.allclose(tensor_merge, dense_diagonal)


def test_merge_noisy_op() -> None:
    ops = [
        pyq.RX(0, "theta_0"),
        pyq.RY(
            0,
            "theta_1",
            noise=DigitalNoiseProtocol(DigitalNoiseType.DEPOLARIZING, 0.1, 0),
        ),
        pyq.RX(0, "theta_2"),
    ]
    circ = pyq.QuantumCircuit(2, ops)
    mergecirc = pyq.Merge(ops)
    state = pyq.random_state(2)
    values = {f"theta_{i}": torch.rand(1) for i in range(3)}
    circ_out = circ(state, values)
    mergecirc_out = mergecirc(state, values)
    assert isinstance(circ_out, DensityMatrix)
    assert isinstance(mergecirc_out, DensityMatrix)
    assert torch.allclose(circ_out, mergecirc_out)


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


def test_sample_run() -> None:
    ops = [pyq.X(0), pyq.X(1)]
    circ = pyq.QuantumCircuit(4, ops)
    wf = run(circ)
    samples = sample(circ)
    assert torch.allclose(wf, product_state("1100"))
    assert torch.allclose(pyq.QuantumCircuit(4, [pyq.I(0)]).run("1100"), wf)
    assert "1100" in samples[0]
