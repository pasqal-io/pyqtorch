from __future__ import annotations

import random
from collections import Counter

import pytest
import torch

import pyqtorch as pyq
from pyqtorch.noise import CorrelatedReadoutNoise, ReadoutNoise
from pyqtorch.noise.readout import (
    WhiteNoise,
    bs_bitflip_corruption,
    create_confusion_matrices,
    create_noise_matrix,
    sample_to_matrix,
)
from pyqtorch.primitives import Primitive
from pyqtorch.utils import sample_multinomial
from pyqtorch.utils_distributions import js_divergence, js_divergence_counters


def test_noise_matrix():
    noise_mat = torch.tensor([[0.5679, 0.7676]])
    zero_prob_mat = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    assert torch.allclose(
        create_confusion_matrices(noise_mat, 0.1), torch.stack([zero_prob_mat] * 2)
    )


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
    corrupted_counters = [bs_bitflip_corruption(err_idx=err_idx, sample=sample)]
    assert sum(corrupted_counters[0].values()) == n_shots
    assert corrupted_counters == exp_corrupted_counters
    assert torch.allclose(
        torch.tensor(1.0 - js_divergence_counters(corrupted_counters[0], counters[0])),
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
    corrupted_counters = [bs_bitflip_corruption(err_idx=err_idx, sample=sample)]
    for noiseless, noisy in zip(counters, corrupted_counters):
        assert sum(noisy.values()) == n_shots
        assert js_divergence_counters(noiseless, noisy) >= 0.0


@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize(
    "error_probability, n_shots, list_ops",
    [
        (0.1, 100, [pyq.X(0), pyq.X(1)]),
    ],
)
def test_readout_error_quantum_circuit(
    error_probability: float,
    n_shots: int,
    list_ops: list[Primitive],
) -> None:

    n_qubits = max([max(op.target) for op in list_ops]) + 1
    noiseless_qc = pyq.QuantumCircuit(n_qubits, list_ops)
    noiseless_samples = noiseless_qc.sample(n_shots=n_shots)

    readout = ReadoutNoise(n_qubits, error_probability=error_probability, seed=0)
    noisy_qc = pyq.QuantumCircuit(n_qubits, list_ops, readout)
    noisy_samples = noisy_qc.sample(n_shots=n_shots)

    for noiseless, noisy in zip(noiseless_samples, noisy_samples):
        assert sum(noiseless.values()) == sum(noisy.values()) == n_shots
        assert js_divergence_counters(noiseless, noisy) > 0.0
        assert torch.allclose(
            torch.tensor(1.0 - js_divergence_counters(noiseless, noisy)),
            torch.ones(1) - error_probability,
            atol=1e-1,
        )


def test_correlated_readout() -> None:
    n_shots = 1000
    confusion_matrix = torch.tensor(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.05, 0.85, 0.05, 0.05],
            [0.03, 0.05, 0.87, 0.05],
            [0.02, 0.05, 0.05, 0.88],
        ],
        dtype=torch.float64,
    )

    corr_readout = CorrelatedReadoutNoise(confusion_matrix, 0)
    probas = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.double)
    out_probas = corr_readout.apply(probas, n_shots=1000)
    assert torch.allclose(
        out_probas,
        torch.tensor([[0.3830, 0.2900, 0.2060, 0.1210]], dtype=torch.float64),
        atol=1e-4,
    )


def test_readout_error_expectation() -> None:

    n_shots = 100
    rx = pyq.RX(0, param_name="theta")
    y = pyq.Y(0)
    cnot = pyq.CNOT(0, 1)
    ops = [rx, y, cnot]
    n_qubits = 2
    noiseless_qc = pyq.QuantumCircuit(n_qubits, ops)
    initstate = pyq.random_state(n_qubits)
    theta = torch.rand(1, requires_grad=True)
    obs = pyq.Observable(pyq.Z(0))

    readout = ReadoutNoise(n_qubits, error_probability=0.1, seed=0)
    noisy_qc = pyq.QuantumCircuit(n_qubits, ops, readout)

    confusion_matrix = torch.tensor(
        [
            [0.9, 0.05, 0.03, 0.02],
            [0.05, 0.85, 0.05, 0.05],
            [0.03, 0.05, 0.87, 0.05],
            [0.02, 0.05, 0.05, 0.88],
        ],
        dtype=torch.float64,
    )
    readoutCorrelated = CorrelatedReadoutNoise(confusion_matrix, seed=0)

    corr_noisy_qc = pyq.QuantumCircuit(n_qubits, ops, readoutCorrelated)
    assert not torch.allclose(
        pyq.expectation(
            noiseless_qc,
            initstate,
            {"theta": theta},
            observable=obs,
        ),
        pyq.expectation(
            noisy_qc, initstate, {"theta": theta}, observable=obs, n_shots=n_shots
        ),
    )
    assert not torch.allclose(
        pyq.expectation(
            noiseless_qc,
            initstate,
            {"theta": theta},
            observable=obs,
        ),
        pyq.expectation(
            corr_noisy_qc, initstate, {"theta": theta}, observable=obs, n_shots=n_shots
        ),
    )


@pytest.mark.flaky(max_runs=5)
def test_readout_apply_probas() -> None:
    n_qubits = 2
    n_shots = 1000
    probas = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.double)
    readobj = ReadoutNoise(2, seed=0)
    out_probas = readobj.apply(probas, n_shots=1000)

    assert torch.allclose(torch.sum(out_probas), torch.ones(1, dtype=out_probas.dtype))
    assert torch.allclose(
        out_probas,
        torch.tensor([[0.3860, 0.2957, 0.2043, 0.1140]], dtype=torch.float64),
        atol=1e-4,
    )

    batch_sample_multinomial = torch.func.vmap(
        lambda p: sample_multinomial(
            p, n_qubits, n_shots, return_counter=False, minlength=probas.shape[-1]
        ),
        randomness="different",
    )

    p = batch_sample_multinomial(probas).squeeze(0) / n_shots
    q = batch_sample_multinomial(out_probas).squeeze(0) / n_shots

    jsd = js_divergence(p, q)
    assert jsd > 0.0

    assert torch.allclose(
        torch.ones(1) - jsd,
        torch.ones(1) - readobj.error_probability,
        atol=1e-1,
    )
