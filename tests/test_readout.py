from __future__ import annotations

import random
from collections import Counter

import pytest
import torch

from pyqtorch import QuantumCircuit, X, expectation, zero_state
from pyqtorch.noise import ReadoutNoise
from pyqtorch.noise.readout import (
    WhiteNoise,
    bs_corruption,
    create_noise_matrix,
    sample_to_matrix,
)
from pyqtorch.primitives import Primitive
from pyqtorch.utils_distributions import js_divergence


@pytest.mark.parametrize(
    "error_probability, counters, exp_corrupted_counters, n_qubits",
    [
        (
            1.0,
            [Counter({"00": 27, "01": 23, "10": 24, "11": 26})],
            [Counter({"11": 27, "10": 23, "01": 24, "00": 26})],
            2,
        ),
        (
            1.0,
            [Counter({"001": 27, "010": 23, "101": 24, "110": 26})],
            [Counter({"110": 27, "101": 23, "010": 24, "001": 26})],
            3,
        ),
    ],
)
def test_bitstring_corruption_all_bitflips(
    error_probability: float,
    counters: list,
    exp_corrupted_counters: list,
    n_qubits: int,
) -> None:
    n_shots = 100
    noise_matrix = create_noise_matrix(WhiteNoise.UNIFORM, n_shots, n_qubits)
    err_idx = torch.as_tensor(noise_matrix < error_probability)
    sample = sample_to_matrix(counters[0])
    corrupted_counters = [bs_corruption(err_idx=err_idx, sample=sample)]
    assert sum(corrupted_counters[0].values()) == n_shots
    assert corrupted_counters == exp_corrupted_counters
    assert torch.allclose(
        torch.tensor(1.0 - js_divergence(corrupted_counters[0], counters[0])),
        torch.ones(1),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "counters, n_qubits",
    [
        (
            [Counter({"00": 27, "01": 23, "10": 24, "11": 26})],
            2,
        ),
        (
            [Counter({"001": 27, "010": 23, "101": 24, "110": 26})],
            3,
        ),
    ],
)
def test_bitstring_corruption_mixed_bitflips(counters: list, n_qubits: int) -> None:
    n_shots = 100
    error_probability = random.random()
    noise_matrix = create_noise_matrix(WhiteNoise.UNIFORM, n_shots, n_qubits)
    err_idx = torch.as_tensor(noise_matrix < error_probability)
    sample = sample_to_matrix(counters[0])
    corrupted_counters = [bs_corruption(err_idx=err_idx, sample=sample)]
    for noiseless, noisy in zip(counters, corrupted_counters):
        assert sum(noisy.values()) == n_shots
        assert js_divergence(noiseless, noisy) >= 0.0


def test_raise_errors_Readout():
    with pytest.raises(AssertionError):
        ReadoutNoise(2, 0.1, noise_matrix=torch.eye(2).unsqueeze(0))

    with pytest.raises(AssertionError):
        ReadoutNoise(2, 0.1, noise_matrix=torch.eye(4))


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "error_probability, n_shots, list_ops",
    [
        (0.1, 100, [X(0), X(1)]),
    ],
)
def test_readout_error_quantum_circuit(
    error_probability: float,
    n_shots: int,
    list_ops: list[Primitive],
) -> None:

    n_qubits = max([max(op.target) for op in list_ops]) + 1
    noiseless_qc = QuantumCircuit(n_qubits, list_ops)
    noiseless_samples = noiseless_qc.sample(n_shots=n_shots)

    readout = ReadoutNoise(n_qubits, error_probability=error_probability, seed=0)
    noisy_qc = QuantumCircuit(n_qubits, list_ops, readout)
    noisy_samples = noisy_qc.sample(n_shots=n_shots)

    for noiseless, noisy in zip(noiseless_samples, noisy_samples):
        assert sum(noiseless.values()) == sum(noisy.values()) == n_shots
        assert js_divergence(noiseless, noisy) > 0.0
        assert torch.allclose(
            torch.tensor(1.0 - js_divergence(noiseless, noisy)),
            torch.ones(1) - error_probability,
            atol=1e-1,
        )

    initstate = zero_state(n_qubits)
    assert expectation(noiseless_qc, initstate, n_shots=n_shots) != expectation(
        noisy_qc, initstate, n_shots=n_shots
    )
