from __future__ import annotations

import random
from typing import Any, Callable

import pytest
import qutip
import torch
from pytest import FixtureRequest
from torch import Tensor

from pyqtorch.apply import apply_operator
from pyqtorch.noise import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    Noise,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
)
from pyqtorch.parametric import PHASE, RX, RY, RZ
from pyqtorch.primitive import H, I, Primitive, S, T, X, Y, Z
from pyqtorch.utils import (
    DensityMatrix,
    density_mat,
    product_state,
    random_dm_promotion,
    random_state,
)


def _calc_mat_vec_wavefunction(
    block: Primitive, n_qubits: int, init_state: torch.Tensor, values: dict = {}
) -> torch.Tensor:
    """Get the result of applying the matrix representation of a block to an initial state.

    Args:
        block: The black operator to apply.
        n_qubits: Number of qubits in the circuit.
        init_state: Initial state to apply block on.
        values: Values of parameter if block is parametric.

    Returns:
        Tensor: The new wavefunction after applying the block.

    """
    mat = block.tensor(n_qubits=len(block.qubit_support), values=values)
    return apply_operator(
        init_state, mat, qubits=block.qubit_support, n_qubits=n_qubits
    )


@pytest.fixture(params=[I, X, Y, Z, H, T, S])
def gate(request: Primitive) -> Any:
    return request.param


# Parametrized fixture
@pytest.fixture
def n_qubits(request: FixtureRequest) -> Any:
    params = getattr(request, "param", {})
    low: int = params.get("low", 1)
    high: int = params.get("high", 6)
    return torch.randint(low=low, high=high, size=(1,)).item()


@pytest.fixture
def target(n_qubits: int) -> int:
    return random.choice([i for i in range(n_qubits)])


# Parametrized fixture
@pytest.fixture
def batch_size(request: FixtureRequest) -> Any:
    params = getattr(request, "param", {})
    low: int = params.get("low", 1)
    high: int = params.get("high", 5)
    return torch.randint(low=low, high=high, size=(1,)).item()


@pytest.fixture
def random_input_state(n_qubits: int, batch_size: int) -> Any:
    return random_state(n_qubits, batch_size)


@pytest.fixture
def rho_input(batch_size: int, target: int, n_qubits: int) -> Any:
    rho_0: DensityMatrix = density_mat(product_state("0", batch_size))
    rho_input: DensityMatrix = random_dm_promotion(target, rho_0, n_qubits)
    return rho_input


@pytest.fixture
def random_gate(target: int) -> Any:
    GATES = [X, Y, Z, I, H]
    gate = random.choice(GATES)
    return gate(target)


@pytest.fixture
def random_rotation_gate(target: int) -> Any:
    ROTATION_GATES = [RX, RY, RZ, PHASE]
    rotation_gate = random.choice(ROTATION_GATES)
    return rotation_gate(target, "theta")


@pytest.fixture
def random_noise_gate(n_qubits: int) -> Any:
    NOISE_GATES = [
        BitFlip,
        PhaseFlip,
        PauliChannel,
        Depolarizing,
        AmplitudeDamping,
        PhaseDamping,
        GeneralizedAmplitudeDamping,
    ]
    noise_gate = random.choice(NOISE_GATES)
    if noise_gate == PauliChannel:
        noise_prob = torch.rand(size=(3,))
        noise_prob = noise_prob / (
            noise_prob.sum(dim=0, keepdim=True) + torch.rand((1,)).item()
        )
        return noise_gate(torch.randint(0, n_qubits, (1,)).item(), noise_prob)
    elif noise_gate == GeneralizedAmplitudeDamping:
        noise_prob = torch.rand(size=(2,))
        p, r = noise_prob[0], noise_prob[1]
        return noise_gate(torch.randint(0, n_qubits, (1,)).item(), p, r)
    else:
        noise_prob = torch.rand(size=(1,)).item()
        return noise_gate(torch.randint(0, n_qubits, (1,)).item(), noise_prob)


@pytest.fixture
def random_flip_gate() -> Any:
    FLIP_GATES = [BitFlip, PhaseFlip, Depolarizing, PauliChannel]
    # The gate is not initialize
    return random.choice(FLIP_GATES)


@pytest.fixture
def flip_probability(random_flip_gate: Noise) -> Any:
    if random_flip_gate == PauliChannel:
        probabilities = torch.rand((3))
        probabilities = probabilities / (
            torch.sum(probabilities) + torch.rand((1,)).item()
        )  # To ensure that the sum of the probabilities is lower than 1.
    else:
        probabilities = torch.rand(1).item()
    return probabilities


@pytest.fixture
def flip_expected_state(
    n_qubits: int,
    target: int,
    random_flip_gate: Noise,
    flip_probability: Tensor,
    batch_size: int,
) -> Any:
    if random_flip_gate == BitFlip:
        expected_state = DensityMatrix(
            torch.tensor(
                [[[1 - flip_probability], [0]], [[0], [flip_probability]]],
                dtype=torch.cdouble,
            )
        )
    elif random_flip_gate == PhaseFlip:
        expected_state = DensityMatrix(
            torch.tensor([[[1], [0]], [[0], [0]]], dtype=torch.cdouble)
        )
    elif random_flip_gate == Depolarizing:
        expected_state = DensityMatrix(
            torch.tensor(
                [
                    [[1 - (2 * flip_probability) / 3], [0]],
                    [[0], [(2 * flip_probability) / 3]],
                ],
                dtype=torch.cdouble,
            )
        )
    elif random_flip_gate == PauliChannel:
        px, py = flip_probability[0], flip_probability[1]
        expected_state = DensityMatrix(
            torch.tensor(
                [[[1 - (px + py)], [0]], [[0], [px + py]]], dtype=torch.cdouble
            )
        )
    expected_state = random_dm_promotion(target, expected_state, n_qubits)
    return expected_state.repeat(1, 1, batch_size)


@pytest.fixture
def flip_gates_prob_0(random_flip_gate: Noise, target: int) -> Any:
    if random_flip_gate == PauliChannel:
        FlipGate_0 = random_flip_gate(target, probabilities=(0, 0, 0))
    else:
        FlipGate_0 = random_flip_gate(target, probability=0)
    return FlipGate_0


@pytest.fixture
def flip_gates_prob_1(
    random_flip_gate: Noise, target: int, random_input_state: Tensor
) -> Any:
    if random_flip_gate == BitFlip:
        FlipGate_1 = random_flip_gate(target, probability=1)
        expected_op = density_mat(X(target)(random_input_state))
    elif random_flip_gate == PhaseFlip:
        FlipGate_1 = random_flip_gate(target, probability=1)
        expected_op = density_mat(Z(target)(random_input_state))
    elif random_flip_gate == Depolarizing:
        FlipGate_1 = random_flip_gate(target, probability=1)
        expected_op = (
            1
            / 3
            * (
                density_mat(Z(target)(random_input_state))
                + density_mat(X(target)(random_input_state))
                + density_mat(Y(target)(random_input_state))
            )
        )
    elif random_flip_gate == PauliChannel:
        px, py, pz = 1 / 3, 1 / 3, 1 / 3
        FlipGate_1 = random_flip_gate(target, probabilities=(px, py, pz))
        expected_op = (
            px * density_mat(X(target)(random_input_state))  # type: ignore
            + py * density_mat(Y(target)(random_input_state))
            + pz * density_mat(Z(target)(random_input_state))
        )
    return FlipGate_1, expected_op


@pytest.fixture
def random_damping_gate() -> Any:
    DAMPING_GATES = [AmplitudeDamping, PhaseDamping, GeneralizedAmplitudeDamping]
    # The gate is not initialize
    return random.choice(DAMPING_GATES)


@pytest.fixture
def damping_rate(random_damping_gate: Noise) -> Any:
    if random_damping_gate == GeneralizedAmplitudeDamping:
        rate = torch.rand((2,))  # prob and rate
    else:
        rate = torch.rand(1).item()
    return rate


@pytest.fixture
def damping_expected_state(
    n_qubits: int,
    target: int,
    random_damping_gate: Noise,
    damping_rate: Tensor,
    batch_size: int,
    rho_input: Tensor,
) -> Any:
    if random_damping_gate == GeneralizedAmplitudeDamping:
        p, r = damping_rate[0], damping_rate[1]
        expected_state = DensityMatrix(
            torch.tensor(
                [[[1 - (r - p * r)], [0]], [[0], [r - p * r]]], dtype=torch.cdouble
            )
        )
        DampingGate = random_damping_gate(target, p, r)
        expected_state = random_dm_promotion(target, expected_state, n_qubits).repeat(
            1, 1, batch_size
        )
    else:
        expected_state = I(target)(rho_input)
        DampingGate = random_damping_gate(target, damping_rate)
    return DampingGate, expected_state


@pytest.fixture
def damping_gates_prob_0(random_damping_gate: Noise, target: int) -> Any:
    if random_damping_gate == GeneralizedAmplitudeDamping:
        damping_gate_0 = random_damping_gate(target, probability=0, rate=0)
    else:
        damping_gate_0 = random_damping_gate(target, rate=0)
    return damping_gate_0


@pytest.fixture
def duration() -> float:
    return float(torch.rand(1))


@pytest.fixture
def n_steps() -> float:
    return int(torch.randint(100, 1000, (1,)))


@pytest.fixture
def omega() -> float:
    return 20.0


@pytest.fixture
def param_x() -> float:
    return float(5.0 * torch.rand(1))


@pytest.fixture
def param_y() -> float:
    return float(2.0 * torch.rand(1))


@pytest.fixture
def sigma_x() -> Tensor:
    return torch.tensor([[0.0, 1.0], [1.0, 0.0]])


@pytest.fixture
def sigma_y() -> Tensor:
    return torch.tensor([[0.0, -1.0j], [1.0j, 0.0]])


@pytest.fixture
def jump_op_torch() -> Tensor:
    return [torch.eye(4, dtype=torch.complex128)]


@pytest.fixture
def jump_op_qutip() -> Tensor:
    return [qutip.qeye(4)]


@pytest.fixture
def torch_hamiltonian(
    omega: float, param_x: float, param_y: float, sigma_x: Tensor, sigma_y: Tensor
) -> Callable:
    def hamiltonian_t(t: float) -> Tensor:
        t = torch.as_tensor(t)
        return omega * (
            param_y * torch.sin(t) * torch.kron(sigma_x, torch.eye(2))
            + param_x * t**2 * torch.kron(torch.eye(2), sigma_y)
        ).to(torch.complex128)

    return hamiltonian_t


@pytest.fixture
def qutip_hamiltonian(omega: float, param_x: float, param_y: float) -> Callable:
    def hamiltonian_t(t: float, args: Any) -> qutip.Qobj:
        return qutip.Qobj(
            omega
            * (
                param_y
                * torch.sin(torch.as_tensor(t)).numpy()
                * qutip.tensor(qutip.sigmax(), qutip.qeye(2))
                + param_x * t**2 * qutip.tensor(qutip.qeye(2), qutip.sigmay())
            ).full()
        )

    return hamiltonian_t
