from __future__ import annotations

import random

import pytest
import torch
from torch import Tensor

from pyqtorch.apply import apply_density_mat, operator_product
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.matrices import (
    HMAT,
    IMAT,
    XMAT,
    YMAT,
    ZMAT,
    _dagger,
)
from pyqtorch.noise import (
    AmplitudeDamping,
    GeneralizedAmplitudeDamping,
    Noise,
    PhaseDamping,
)
from pyqtorch.primitives import (
    ControlledPrimitive,
    ControlledRotationGate,
    H,
    I,
    Parametric,
    Primitive,
    X,
    Y,
    Z,
)
from pyqtorch.utils import (
    DensityMatrix,
    density_mat,
    operator_kron,
    product_state,
    random_state,
)


def test_dm(n_qubits: int, batch_size: int) -> None:
    state = random_state(n_qubits)
    projector = torch.outer(state.flatten(), state.conj().flatten()).view(
        2**n_qubits, 2**n_qubits, 1
    )
    dm = density_mat(state)
    assert dm.size() == torch.Size([2**n_qubits, 2**n_qubits, 1])
    assert torch.allclose(dm, projector)
    assert torch.allclose(dm.squeeze(), dm.squeeze() @ dm.squeeze())
    states = []
    projectors = []
    for batch in range(batch_size):
        state = random_state(n_qubits)
        states.append(state)
        projector = torch.outer(state.flatten(), state.conj().flatten()).view(
            2**n_qubits, 2**n_qubits, 1
        )
        projectors.append(projector)
    dm_proj = torch.cat(projectors, dim=2)
    state_cat = torch.cat(states, dim=n_qubits)
    dm = density_mat(state_cat)
    assert dm.size() == torch.Size([2**n_qubits, 2**n_qubits, batch_size])
    assert torch.allclose(dm, dm_proj)


@pytest.mark.parametrize("n_qubits", [{"low": 2, "high": 5}], indirect=True)
def test_operator_product(
    random_unitary_gate: Primitive | Parametric,
    n_qubits: int,
) -> None:
    batch_size_1 = torch.randint(low=1, high=5, size=(1,)).item()
    batch_size_2 = torch.randint(low=1, high=5, size=(1,)).item()
    max_batch = max(batch_size_2, batch_size_1)
    values = {"theta": torch.rand(1)}
    op = random_unitary_gate.tensor(values=values, full_support=tuple(range(n_qubits)))
    op_mul = operator_product(
        op.repeat(1, 1, batch_size_1), _dagger(op.repeat(1, 1, batch_size_2))
    )
    assert op_mul.size() == torch.Size([2**n_qubits, 2**n_qubits, max_batch])
    assert torch.allclose(
        op_mul,
        torch.eye(2**n_qubits, dtype=torch.cdouble)
        .unsqueeze(2)
        .repeat(1, 1, max_batch),
    )


@pytest.mark.parametrize("n_qubits", [{"low": 2, "high": 5}], indirect=True)
def test_apply_density_mat(
    random_unitary_gate: Primitive | Parametric,
    n_qubits: int,
    batch_size: int,
    random_input_dm: DensityMatrix,
) -> None:
    values = {"theta": torch.rand(1)}
    op = random_unitary_gate.tensor(values=values, full_support=tuple(range(n_qubits)))
    rho = random_input_dm
    rho_evol = apply_density_mat(op, rho)
    assert rho_evol.size() == torch.Size([2**n_qubits, 2**n_qubits, batch_size])
    rho_expected = operator_product(op, operator_product(rho, _dagger(op)))
    assert torch.allclose(rho_evol, rho_expected)


@pytest.mark.parametrize(
    "operator,matrix", [(I, IMAT), (X, XMAT), (Z, ZMAT), (Y, YMAT), (H, HMAT)]
)
def test_operator_kron(operator: Tensor, matrix: Tensor) -> None:
    n_qubits = torch.randint(low=1, high=5, size=(1,)).item()
    batch_size = torch.randint(low=1, high=5, size=(1,)).item()
    states, krons = [], []
    for batch in range(batch_size):
        state = random_state(n_qubits)
        states.append(state)
        kron = torch.kron(density_mat(state).squeeze(), matrix).unsqueeze(2)
        krons.append(kron)
    input_state = torch.cat(states, dim=n_qubits)
    kron_out = operator_kron(density_mat(input_state), operator(0).dagger())
    assert kron_out.size() == torch.Size(
        [2 ** (n_qubits + 1), 2 ** (n_qubits + 1), batch_size]
    )
    kron_expect = torch.cat(krons, dim=2)
    assert torch.allclose(kron_out, kron_expect)
    assert torch.allclose(
        torch.kron(operator(0).dagger().contiguous(), I(0).tensor()),
        operator_kron(operator(0).dagger(), I(0).tensor()),
    )


def test_kron_batch() -> None:
    n_qubits = torch.randint(low=1, high=5, size=(1,)).item()
    batch_size_1 = torch.randint(low=1, high=5, size=(1,)).item()
    batch_size_2 = torch.randint(low=1, high=5, size=(1,)).item()
    max_batch = max(batch_size_1, batch_size_2)
    dm_1 = density_mat(random_state(n_qubits, batch_size_1))
    dm_2 = density_mat(random_state(n_qubits, batch_size_2))
    dm_out = operator_kron(dm_1, dm_2)
    assert dm_out.size() == torch.Size(
        [2 ** (2 * n_qubits), 2 ** (2 * n_qubits), max_batch]
    )
    if batch_size_1 > batch_size_2:
        dm_2 = dm_2.repeat(1, 1, batch_size_1)[:, :, :batch_size_1]
    elif batch_size_2 > batch_size_1:
        dm_1 = dm_1.repeat(1, 1, batch_size_2)[:, :, :batch_size_2]
    density_matrices = []
    for batch in range(max_batch):
        density_matrice = torch.kron(dm_1[:, :, batch], dm_2[:, :, batch]).unsqueeze(2)
        density_matrices.append(density_matrice)
    dm_expect = torch.cat(density_matrices, dim=2)
    assert torch.allclose(dm_out, dm_expect)


@pytest.mark.parametrize("n_qubits", [{"low": 2, "high": 5}], indirect=True)
def test_flip_gates(
    n_qubits: int,
    target: int,
    batch_size: int,
    rho_input: Tensor,
    random_flip_gate: Noise,
    flip_expected_state: DensityMatrix,
    flip_probability: Tensor | float,
    flip_gates_prob_0: Noise,
    flip_gates_prob_1: tuple,
    random_input_dm: DensityMatrix,
) -> None:
    FlipGate = random_flip_gate
    output_state: DensityMatrix = FlipGate(target, flip_probability)(rho_input)
    assert output_state.size() == torch.Size([2**n_qubits, 2**n_qubits, batch_size])
    assert torch.allclose(output_state, flip_expected_state)

    input_state = random_input_dm  # fix the same random state for every call
    assert torch.allclose(flip_gates_prob_0(input_state), input_state)

    FlipGate_1, expected_op = flip_gates_prob_1
    assert torch.allclose(FlipGate_1(input_state), expected_op)


def test_damping_gates(
    n_qubits: int,
    target: int,
    batch_size: int,
    damping_expected_state: tuple,
    damping_gates_prob_0: Tensor,
    random_input_dm: DensityMatrix,
    rho_input: Tensor,
) -> None:
    DampingGate, expected_state = damping_expected_state
    apply_gate = DampingGate(rho_input)
    assert apply_gate.size() == torch.Size([2**n_qubits, 2**n_qubits, batch_size])
    assert torch.allclose(apply_gate, expected_state)

    input_state = random_input_dm
    assert torch.allclose(
        damping_gates_prob_0(input_state),
        I(target)(input_state),
    )

    rho_0: DensityMatrix = density_mat(product_state("0", batch_size))
    rho_1: DensityMatrix = density_mat(product_state("1", batch_size))
    if DampingGate == AmplitudeDamping:
        assert torch.allclose(DampingGate(target, rate=1)(rho_1), rho_0)
    elif DampingGate == PhaseDamping:
        assert torch.allclose(DampingGate(target, rate=1)(rho_1), I(target)(rho_1))
    elif DampingGate == GeneralizedAmplitudeDamping:
        assert torch.allclose(
            DampingGate(target, error_probability=(1, 1))(rho_1), rho_0
        )


@pytest.mark.parametrize("n_qubits", [{"low": 2, "high": 5}], indirect=True)
def test_noisy_primitive(
    random_noisy_unitary_gate: tuple,
    random_input_dm: DensityMatrix,
    n_qubits: int,
    batch_size: int,
) -> None:
    noisy_primitive, primitve_gate, noise_gate = random_noisy_unitary_gate
    state = random_input_dm
    values = {"theta": torch.rand(1)}
    rho_evol = noisy_primitive(state, values)
    assert rho_evol.size() == torch.Size([2**n_qubits, 2**n_qubits, batch_size])
    rho_expected = noise_gate(primitve_gate(state, values))
    assert rho_expected.size() == torch.Size([2**n_qubits, 2**n_qubits, batch_size])
    assert torch.allclose(rho_evol, rho_expected)


@pytest.mark.parametrize("n_qubits", [{"low": 2, "high": 5}], indirect=True)
@pytest.mark.parametrize("batch_size", [{"low": 1, "high": 5}], indirect=True)
def test_noise_circ(
    n_qubits: int,
    batch_size: int,
    random_input_dm: DensityMatrix,
    random_single_qubit_gate: Primitive,
    random_noise_gate: Noise,
    random_rotation_gate: Parametric,
    random_controlled_gate: ControlledPrimitive,
    random_rotation_control_gate: ControlledRotationGate,
) -> None:
    OPERATORS = [
        random_single_qubit_gate,
        random_noise_gate,
        random_rotation_gate,
        random_controlled_gate,
        random_rotation_control_gate,
    ]
    random.shuffle(OPERATORS)
    circ = QuantumCircuit(n_qubits, OPERATORS)

    values = {random_rotation_gate.param_name: torch.rand(1)}
    output_state = circ(random_input_dm, values)
    assert isinstance(output_state, DensityMatrix)
    assert output_state.shape == torch.Size([2**n_qubits, 2**n_qubits, batch_size])

    diag_sums = []
    for i in range(batch_size):
        diag_batch = torch.diagonal(output_state[:, :, i], dim1=0, dim2=1)
        diag_sums.append(torch.sum(diag_batch))
    diag_sum = torch.stack(diag_sums)
    assert torch.allclose(diag_sum, torch.ones((batch_size,), dtype=torch.cdouble))
