from __future__ import annotations

import random

import pytest
import torch
from torch import Tensor

import pyqtorch as pyq
from pyqtorch import DiffMode, expectation, run, sample
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.matrices import COMPLEX_TO_REAL_DTYPES
from pyqtorch.noise import Noise
from pyqtorch.parametric import Parametric
from pyqtorch.primitive import Primitive
from pyqtorch.utils import GRADCHECK_ATOL, DensityMatrix, product_state


def test_adjoint_diff() -> None:
    rx = pyq.RX(0, param_name="theta_0")
    cry = pyq.CPHASE(0, 1, param_name="theta_1")
    rz = pyq.RZ(2, param_name="theta_2")
    cnot = pyq.CNOT(1, 2)
    ops = [rx, cry, rz, cnot]
    n_qubits = 3
    circ = pyq.QuantumCircuit(n_qubits, ops)
    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])

    theta_0_value = torch.pi / 2
    theta_1_value = torch.pi
    theta_2_value = torch.pi / 4

    state = pyq.zero_state(n_qubits)

    theta_0_ad = torch.tensor([theta_0_value], requires_grad=True)
    thetas_0_adjoint = torch.tensor([theta_0_value], requires_grad=True)

    theta_1_ad = torch.tensor([theta_1_value], requires_grad=True)
    thetas_1_adjoint = torch.tensor([theta_1_value], requires_grad=True)

    theta_2_ad = torch.tensor([theta_2_value], requires_grad=True)
    thetas_2_adjoint = torch.tensor([theta_2_value], requires_grad=True)

    values_ad = {"theta_0": theta_0_ad, "theta_1": theta_1_ad, "theta_2": theta_2_ad}
    values_adjoint = {
        "theta_0": thetas_0_adjoint,
        "theta_1": thetas_1_adjoint,
        "theta_2": thetas_2_adjoint,
    }
    exp_ad = expectation(circ, state, values_ad, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, state, values_adjoint, obs, DiffMode.ADJOINT)

    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad)
    )

    grad_adjoint = torch.autograd.grad(
        exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
    )

    assert len(grad_ad) == len(grad_adjoint)
    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_adjoint[i])


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("n_qubits", [3, 4])
def test_differentiate_circuit(
    dtype: torch.dtype, batch_size: int, n_qubits: int
) -> None:
    ops = [
        pyq.Y(1),
        pyq.RX(0, "theta_0"),
        pyq.PHASE(0, "theta_1"),
        pyq.CSWAP(0, (1, 2)),
        pyq.CRX(1, 2, "theta_2"),
        pyq.CPHASE(1, 2, "theta_3"),
        pyq.CNOT(0, 1),
        pyq.Toffoli((2, 1), 0),
    ]
    theta_0_value = torch.rand(1, dtype=dtype)
    theta_1_value = torch.rand(1, dtype=dtype)
    theta_2_value = torch.rand(1, dtype=dtype)
    theta_3_value = torch.rand(1, dtype=dtype)
    circ = pyq.QuantumCircuit(n_qubits, ops).to(dtype)
    state = pyq.random_state(n_qubits, batch_size, dtype=dtype)

    theta_0_ad = torch.tensor([theta_0_value], requires_grad=True)
    theta_0_adjoint = torch.tensor([theta_0_value], requires_grad=True)

    theta_1_ad = torch.tensor([theta_1_value], requires_grad=True)
    theta_1_adjoint = torch.tensor([theta_1_value], requires_grad=True)

    theta_2_ad = torch.tensor([theta_2_value], requires_grad=True)
    theta_2_adjoint = torch.tensor([theta_2_value], requires_grad=True)

    theta_3_ad = torch.tensor([theta_3_value], requires_grad=True)
    theta_3_adjoint = torch.tensor([theta_3_value], requires_grad=True)

    values_ad = torch.nn.ParameterDict(
        {
            "theta_0": theta_0_ad,
            "theta_1": theta_1_ad,
            "theta_2": theta_2_ad,
            "theta_3": theta_3_ad,
        }
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])
    values_adjoint = torch.nn.ParameterDict(
        {
            "theta_0": theta_0_adjoint,
            "theta_1": theta_1_adjoint,
            "theta_2": theta_2_adjoint,
            "theta_3": theta_3_adjoint,
        }
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])

    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)]).to(dtype)
    exp_ad = expectation(circ, state, values_ad, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, state, values_adjoint, obs, DiffMode.ADJOINT)

    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad)
    )

    grad_adjoint = torch.autograd.grad(
        exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
    )

    assert len(grad_ad) == len(grad_adjoint)
    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_adjoint[i], atol=GRADCHECK_ATOL)


def test_device_inference() -> None:
    ops = [pyq.RX(0), pyq.RX(0)]
    circ = pyq.QuantumCircuit(2, ops)
    nested_circ = pyq.QuantumCircuit(2, [circ, circ])
    assert nested_circ._device is not None


def test_adjoint_duplicate_params() -> None:
    n_qubits = 2
    ops = [pyq.RX(0, param_name="theta_0"), pyq.RX(0, param_name="theta_0")]
    theta_vals = torch.arange(0, torch.pi, 0.05, requires_grad=True)
    circ = pyq.QuantumCircuit(n_qubits, ops)
    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)])
    init_state = pyq.zero_state(n_qubits)
    values = {"theta_0": theta_vals}
    exp_ad = expectation(circ, init_state, values, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, init_state, values, obs, DiffMode.ADJOINT)
    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values.values()), torch.ones_like(exp_ad)
    )[0]
    grad_adjoint = torch.autograd.grad(
        exp_adjoint, tuple(values.values()), torch.ones_like(exp_adjoint)
    )[0]
    assert torch.allclose(grad_ad, grad_adjoint)


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


@pytest.mark.xfail  # investigate
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("n_qubits", [2])
def test_adjoint_scale(dtype: torch.dtype, batch_size: int, n_qubits: int) -> None:
    ops = [pyq.Scale(pyq.X(0), "theta_4")]

    theta_4_value = torch.rand(1, dtype=dtype)
    circ = pyq.QuantumCircuit(n_qubits, ops).to(dtype)

    state = pyq.random_state(n_qubits, batch_size, dtype=dtype)

    theta_4_ad = torch.tensor([theta_4_value], requires_grad=True)
    theta_4_adjoint = torch.tensor([theta_4_value], requires_grad=True)

    values_ad = torch.nn.ParameterDict(
        {
            "theta_4": theta_4_ad,
        }
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])
    values_adjoint = torch.nn.ParameterDict(
        {
            "theta_4": theta_4_adjoint,
        }
    ).to(COMPLEX_TO_REAL_DTYPES[dtype])

    obs = pyq.QuantumCircuit(n_qubits, [pyq.Z(0)]).to(dtype)
    exp_ad = expectation(circ, state, values_ad, obs, DiffMode.AD)
    exp_adjoint = expectation(circ, state, values_adjoint, obs, DiffMode.ADJOINT)

    grad_ad = torch.autograd.grad(
        exp_ad, tuple(values_ad.values()), torch.ones_like(exp_ad)
    )

    grad_adjoint = torch.autograd.grad(
        exp_adjoint, tuple(values_adjoint.values()), torch.ones_like(exp_adjoint)
    )

    assert len(grad_ad) == len(grad_adjoint)
    for i in range(len(grad_ad)):
        assert torch.allclose(grad_ad[i], grad_adjoint[i], atol=GRADCHECK_ATOL)


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
