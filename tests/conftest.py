from __future__ import annotations

import random
from typing import Any

import pytest
import torch
from pytest import FixtureRequest

from pyqtorch.noise import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
)
from pyqtorch.primitive import H, I, Primitive, X, Y, Z

# Parametrized fixture
@pytest.fixture
def n_qubits(request: FixtureRequest) -> Any:
    low: int = request.param.get("low", 1)
    high: int = request.param.get("high", 6)
    return torch.randint(low=low, high=high, size=(1,)).item()


# Parametrized fixture
@pytest.fixture
def batch_size(request: FixtureRequest) -> Any:
    low: int = request.param.get("low", 1)
    high: int = request.param.get("high", 5)
    return torch.randint(low=low, high=high, size=(1,)).item()


@pytest.fixture
def random_gate(n_qubits: int) -> Any:
    GATES = [X, Y, Z, I, H]
    gate = random.choice(GATES)
    return gate(torch.randint(0, n_qubits, (1,)).item())


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
