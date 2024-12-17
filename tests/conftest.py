from __future__ import annotations

import random
from typing import Any, Callable

import pytest
import qutip
import torch
from pytest import FixtureRequest
from torch import Tensor

from pyqtorch.composite import Add, Scale, Sequence
from pyqtorch.embed import ConcretizedCallable
from pyqtorch.noise import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    DigitalNoiseProtocol,
    DigitalNoiseType,
    GeneralizedAmplitudeDamping,
    Noise,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
)
from pyqtorch.primitives import (
    CNOT,
    CPHASE,
    CRX,
    CRY,
    CRZ,
    CY,
    CZ,
    PHASE,
    RX,
    RY,
    RZ,
    ControlledPrimitive,
    ControlledRotationGate,
    H,
    I,
    Parametric,
    Primitive,
    S,
    T,
    X,
    Y,
    Z,
)
from pyqtorch.utils import (
    DensityMatrix,
    density_mat,
    product_state,
    random_dm_promotion,
    random_state,
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
def random_input_dm(random_input_state: Tensor) -> Any:
    return density_mat(random_input_state)


@pytest.fixture
def rho_input(batch_size: int, target: int, n_qubits: int) -> Any:
    rho_0: DensityMatrix = density_mat(product_state("0", batch_size))
    rho_input: DensityMatrix = random_dm_promotion(target, rho_0, n_qubits)
    return rho_input


# TODO: create random noisy protocols
@pytest.fixture
def random_noise():
    pass


@pytest.fixture
def random_single_qubit_gate(target: int) -> Any:
    GATES = [X, Y, Z, I, H]
    gate = random.choice(GATES)
    return gate(target)


@pytest.fixture
def random_rotation_gate(target: int) -> Any:
    ROTATION_GATES = [RX, RY, RZ, PHASE]
    rotation_gate = random.choice(ROTATION_GATES)
    return rotation_gate(target, "theta")


@pytest.fixture
def random_controlled_gate(n_qubits: int, target: int) -> Any:
    if n_qubits < 2:
        raise ValueError("The controlled gates are defined on 2 qubits minimum")
    CONTROLLED_GATES = [CNOT, CY, CZ]
    controlled_gate = random.choice(CONTROLLED_GATES)
    control = random.choice([i for i in range(n_qubits) if i != target])
    return controlled_gate(control, target)


@pytest.fixture
def random_rotation_control_gate(n_qubits: int, target: int) -> Any:
    if n_qubits < 2:
        raise ValueError("The controlled gates are defined on 2 qubits minimum")
    ROTATION_CONTROL_GATES = [CRX, CRY, CRZ, CPHASE]
    controlled_gate = random.choice(ROTATION_CONTROL_GATES)
    control = random.choice([i for i in range(n_qubits) if i != target])
    return controlled_gate(control, target, "theta")


@pytest.fixture
def random_unitary_gate(
    random_single_qubit_gate: Primitive,
    random_rotation_gate: Parametric,
    random_controlled_gate: ControlledPrimitive,
    random_rotation_control_gate: ControlledRotationGate,
) -> Any:
    UNITARY_GATES = [
        random_single_qubit_gate,
        random_controlled_gate,
        random_rotation_gate,
        random_rotation_control_gate,
    ]
    unitary_gate = random.choice(UNITARY_GATES)
    return unitary_gate


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
        return noise_gate(torch.randint(0, n_qubits, (1,)).item(), noise_prob)
    else:
        noise_prob = torch.rand(size=(1,)).item()
        return noise_gate(torch.randint(0, n_qubits, (1,)).item(), noise_prob)


@pytest.fixture
def random_noisy_protocol() -> Any:
    noise_type = random.choice(list(DigitalNoiseType))
    if noise_type == DigitalNoiseType.PAULI_CHANNEL:
        noise_prob = torch.rand(size=(3,))
        noise_prob = noise_prob / (
            noise_prob.sum(dim=0, keepdim=True) + torch.rand((1,)).item()
        )
    elif noise_type == DigitalNoiseType.GENERALIZED_AMPLITUDE_DAMPING:
        noise_prob = torch.rand(size=(2,))
    else:
        noise_prob = torch.rand(size=(1,)).item()
    return DigitalNoiseProtocol(noise_type, noise_prob)


@pytest.fixture
def random_noisy_unitary_gate(
    n_qubits: int, target: int, random_noisy_protocol: DigitalNoiseProtocol
) -> Any:
    if n_qubits < 2:
        raise ValueError("The controlled gates are defined on 2 qubits minimum")
    SINGLE_GATES = [X, Y, Z, I, H]
    ROTATION_GATES = [RX, RY, RZ, PHASE]
    CONTROLLED_GATES = [CNOT, CY, CZ]
    ROTATION_CONTROL_GATES = [CRX, CRY, CRZ, CPHASE]
    UNITARY_GATES = (
        SINGLE_GATES + ROTATION_GATES + CONTROLLED_GATES + ROTATION_CONTROL_GATES
    )
    unitary_gate = random.choice(UNITARY_GATES)
    protocol_gates, protocol_info = random_noisy_protocol.gates[0]
    noise_gate = protocol_gates(target, protocol_info.error_probability)
    if unitary_gate in SINGLE_GATES:
        return (
            unitary_gate(target=target, noise=random_noisy_protocol),  # type: ignore[call-arg]
            unitary_gate(target=target),  # type: ignore[call-arg]
            noise_gate,
        )  # type: ignore[call-arg]
    if unitary_gate in ROTATION_GATES:
        return (
            unitary_gate(
                target=target, param_name="theta", noise=random_noisy_protocol
            ),  # type: ignore[call-arg]
            unitary_gate(target=target, param_name="theta"),  # type: ignore[call-arg]
            noise_gate,
        )
    if unitary_gate in CONTROLLED_GATES:
        control = random.choice([i for i in range(n_qubits) if i != target])
        return (
            unitary_gate(control=control, target=target, noise=random_noisy_protocol),  # type: ignore[call-arg]
            unitary_gate(control=control, target=target),  # type: ignore[call-arg]
            noise_gate,
        )
    if unitary_gate in ROTATION_CONTROL_GATES:
        control = random.choice([i for i in range(n_qubits) if i != target])
        return (
            unitary_gate(
                control=control,
                target=target,
                param_name="theta",
                noise=random_noisy_protocol,
            ),  # type: ignore[call-arg]
            unitary_gate(control=control, target=target, param_name="theta"),  # type: ignore[call-arg]
            noise_gate,
        )


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
        FlipGate_0 = random_flip_gate(target, error_probability=(0, 0, 0))
    else:
        FlipGate_0 = random_flip_gate(target, error_probability=0)
    return FlipGate_0


@pytest.fixture
def flip_gates_prob_1(
    random_flip_gate: Noise, target: int, random_input_state: Tensor
) -> Any:
    if random_flip_gate == BitFlip:
        FlipGate_1 = random_flip_gate(target, error_probability=1)
        expected_op = density_mat(X(target)(random_input_state))
    elif random_flip_gate == PhaseFlip:
        FlipGate_1 = random_flip_gate(target, error_probability=1)
        expected_op = density_mat(Z(target)(random_input_state))
    elif random_flip_gate == Depolarizing:
        FlipGate_1 = random_flip_gate(target, error_probability=1)
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
        FlipGate_1 = random_flip_gate(target, error_probability=(px, py, pz))
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
        DampingGate = random_damping_gate(target, damping_rate)
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
        damping_gate_0 = random_damping_gate(target, (0, 0))
    else:
        damping_gate_0 = random_damping_gate(target, 0)
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


@pytest.fixture
def tparam() -> str:
    return "t"


@pytest.fixture
def sin(tparam: str) -> tuple[str, ConcretizedCallable]:
    sin_t, sin_fn = "sin_x", ConcretizedCallable("sin", [tparam])
    return sin_t, sin_fn


@pytest.fixture
def sq(tparam: str) -> tuple[str, ConcretizedCallable]:
    t_sq, t_sq_fn = "t_sq", ConcretizedCallable("mul", [tparam, tparam])
    return t_sq, t_sq_fn


@pytest.fixture
def hamevo_generator(
    omega: float, param_x: float, param_y: float, sin: tuple, sq: tuple
) -> Sequence:
    sin_t, _ = sin
    t_sq, _ = sq
    generator = Scale(
        Add(
            [
                Scale(Scale(X(0), sin_t), "y"),
                Scale(Scale(Y(1), t_sq), param_x),
            ]
        ),
        omega,
    )
    return generator
