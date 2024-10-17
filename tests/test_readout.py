from __future__ import annotations

from collections import Counter

import pytest
import torch

from pyqtorch.noise.readout import (
    WhiteNoise,
    bs_corruption,
    create_noise_matrix,
    sample_to_matrix,
)
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
