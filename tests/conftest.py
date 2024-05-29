from __future__ import annotations

import random
from typing import Any

import pytest
import torch
from pytest import FixtureRequest
from torch import Tensor

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
from pyqtorch.primitive import H, I, X, Y, Z
from pyqtorch.utils import DensityMatrix, density_mat, random_dm_promotion, random_state


# Parametrized fixture
@pytest.fixture
def n_qubits(request: FixtureRequest) -> Any:
    low: int = request.param.get("low", 1)
    high: int = request.param.get("high", 6)
    return torch.randint(low=low, high=high, size=(1,)).item()


@pytest.fixture
def target(n_qubits: int) -> int:
    return random.choice([i for i in range(n_qubits)])


# Parametrized fixture
@pytest.fixture
def batch_size(request: FixtureRequest) -> Any:
    low: int = request.param.get("low", 1)
    high: int = request.param.get("high", 5)
    return torch.randint(low=low, high=high, size=(1,)).item()


@pytest.fixture
def random_input_state(n_qubits: int, batch_size: int) -> Any:
    return random_state(n_qubits, batch_size)


@pytest.fixture
def random_gate(n_qubits: int, target: int) -> Any:
    GATES = [X, Y, Z, I, H]
    gate = random.choice(GATES)
    return gate(target)


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
        noise_prob = noise_prob / (noise_prob.sum(dim=0, keepdim=True) + torch.rand((1,)).item())
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
    n_qubits: int, target: int, random_flip_gate: Noise, flip_probability: Tensor, batch_size: int
) -> Any:
    if random_flip_gate == BitFlip:
        expected_state = DensityMatrix(
            torch.tensor(
                [[[1 - flip_probability], [0]], [[0], [flip_probability]]], dtype=torch.cdouble
            )
        )
    elif random_flip_gate == PhaseFlip:
        expected_state = DensityMatrix(torch.tensor([[[1], [0]], [[0], [0]]], dtype=torch.cdouble))
    elif random_flip_gate == Depolarizing:
        expected_state = DensityMatrix(
            torch.tensor(
                [[[1 - (2 * flip_probability) / 3], [0]], [[0], [(2 * flip_probability) / 3]]],
                dtype=torch.cdouble,
            )
        )
    elif random_flip_gate == PauliChannel:
        px, py = flip_probability[0], flip_probability[1]
        expected_state = DensityMatrix(
            torch.tensor([[[1 - (px + py)], [0]], [[0], [px + py]]], dtype=torch.cdouble)
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
def flip_gates_prob_1(random_flip_gate: Noise, target: int, random_input_state: Tensor) -> Any:
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
            px * density_mat(X(target)(random_input_state))
            + py * density_mat(Y(target)(random_input_state))
            + pz * density_mat(Z(target)(random_input_state))
        )
    return FlipGate_1, expected_op
