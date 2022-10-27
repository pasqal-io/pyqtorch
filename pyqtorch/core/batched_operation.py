# Copyright 2022 PyQ Development Team

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from pyqtorch.converters.store_ops import ops_cache, store_operation
from pyqtorch.core.operation import RX, H
from pyqtorch.core.utils import _apply_batch_gate

IMAT = torch.eye(2, dtype=torch.cdouble)
XMAT = torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble)
YMAT = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble)
ZMAT = torch.tensor([[1, 0], [0, -1]], dtype=torch.cdouble)


def batchedRX(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:

    if ops_cache.enabled:
        store_operation("RX", qubits, param=theta)

    dev = state.device
    batch_size = len(theta)

    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    imat = IMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)
    xmat = XMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)

    mat = cos_t * imat - 1j * sin_t * xmat

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)


def batchedRY(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:

    if ops_cache.enabled:
        store_operation("RY", qubits, param=theta)

    dev = state.device
    batch_size = len(theta)

    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    imat = IMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)
    xmat = YMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)

    mat = cos_t * imat - 1j * sin_t * xmat

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)


def batchedRZ(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:

    if ops_cache.enabled:
        store_operation("RZ", qubits, param=theta)

    dev = state.device
    batch_size = len(theta)

    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    imat = IMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)
    xmat = ZMAT.unsqueeze(2).repeat(1, 1, batch_size).to(dev)

    mat = cos_t * imat - 1j * sin_t * xmat

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)


def batchedRZZ(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:

    if ops_cache.enabled:
        store_operation("RZZ", qubits, param=theta)

    dev = state.device
    batch_size = len(theta)

    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((4, 4, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((4, 4, 1))

    mat = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.cdouble).to(dev))

    imat = (
        torch.eye(4, dtype=torch.cdouble).unsqueeze(2).repeat(1, 1, batch_size).to(dev)
    )
    xmat = mat.unsqueeze(2).repeat(1, 1, batch_size).to(dev)

    mat = cos_t * imat + 1j * sin_t * xmat

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)


def batchedRXX(
    theta: torch.Tensor, state: torch.Tensor, qubits: Any, N_qubits: int
) -> torch.Tensor:

    if ops_cache.enabled:
        store_operation("RXX", qubits, param=theta)

    for q in qubits:
        state = H(state, [q], N_qubits)
    state = batchedRZZ(theta, state, qubits, N_qubits)
    for q in qubits:
        state = H(state, [q], N_qubits)

    return state


def batchedRYY(
    theta: torch.Tensor, state: torch.Tensor, qubits: Any, N_qubits: int
) -> torch.Tensor:

    if ops_cache.enabled:
        store_operation("RYY", qubits, param=theta)

    for q in qubits:
        state = RX(torch.tensor(np.pi / 2), state, [q], N_qubits)
    state = batchedRZZ(theta, state, qubits, N_qubits)
    for q in qubits:
        state = RX(-torch.tensor(np.pi / 2), state, [q], N_qubits)

    return state
