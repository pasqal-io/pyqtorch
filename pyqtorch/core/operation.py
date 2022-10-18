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

from typing import List

import torch
from torch import Tensor
import numpy as np

from pyqtorch.core.utils import _apply_gate, _apply_batch_gate
from pyqtorch.converters.store_ops import storable

IMAT = torch.eye(2, dtype=torch.cdouble)
XMAT = torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble)
YMAT = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble)
ZMAT = torch.tensor([[1, 0], [0, -1]], dtype=torch.cdouble)


@storable
def RX(theta: float, state: Tensor, qubits: List[int], N_qubits: int):
    dev = state.device
    mat = IMAT.to(dev) * torch.cos(theta / 2) - 1j * XMAT.to(dev) * torch.sin(theta / 2)
    return _apply_gate(state, mat, qubits, N_qubits)


@storable
def RY(theta: float, state: Tensor, qubits: List[int], N_qubits: int):
    dev = state.device
    mat = IMAT.to(dev) * torch.cos(theta / 2) - 1j * YMAT.to(dev) * torch.sin(theta / 2)
    return _apply_gate(state, mat, qubits, N_qubits)


@storable
def RZ(theta: float, state: Tensor, qubits: List[int], N_qubits: int):
    dev = state.device
    mat = IMAT.to(dev) * torch.cos(theta / 2) + 1j * ZMAT.to(dev) * torch.sin(theta / 2)
    return _apply_gate(state, mat, qubits, N_qubits)


@storable
def RZZ(theta, state, qubits, N_qubits):
    dev = state.device
    mat = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.cdouble).to(dev))
    mat = 1j * torch.sin(theta / 2) * mat + torch.cos(theta / 2) * torch.eye(
        4, dtype=torch.cdouble
    ).to(dev)
    return _apply_gate(state, torch.diag(mat), qubits, N_qubits)


def U(phi, theta, omega, state, qubits, N_qubits):
    """U(phi, theta, omega) = RZ(omega)RY(theta)RZ(phi)"""
    dev = state.device
    t_plus = torch.exp(-1j * (phi + omega) / 2)
    t_minus = torch.exp(-1j * (phi - omega) / 2)
    mat = (
        torch.tensor([[1, 0], [0, 0]], dtype=torch.cdouble).to(dev)
        * torch.cos(theta / 2)
        * t_plus
        - torch.tensor([[0, 1], [0, 0]], dtype=torch.cdouble).to(dev)
        * torch.sin(theta / 2)
        * torch.conj(t_minus)
        + torch.tensor([[0, 0], [1, 0]], dtype=torch.cdouble).to(dev)
        * torch.sin(theta / 2)
        * t_minus
        + torch.tensor([[0, 0], [0, 1]], dtype=torch.cdouble).to(dev)
        * torch.cos(theta / 2)
        * torch.conj(t_plus)
    )
    return _apply_gate(state, mat, qubits, N_qubits)


@storable
def X(state, qubits, N_qubits):
    dev = state.device
    mat = XMAT.to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


@storable
def Z(state, qubits, N_qubits):
    dev = state.device
    mat = ZMAT.to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


@storable
def Y(state, qubits, N_qubits):
    dev = state.device
    mat = YMAT.to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


@storable
def H(state, qubits, N_qubits):
    dev = state.device
    mat = (
        1
        / torch.sqrt(torch.tensor(2))
        * torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble).to(dev)
    )
    return _apply_gate(state, mat, qubits, N_qubits)


@storable
def CNOT(state, qubits, N_qubits):
    dev = state.device
    mat = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.cdouble
    ).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


@storable
def batchedRX(theta, state, qubits, N_qubits):
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


@storable
def batchedRY(theta, state, qubits, N_qubits):
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


@storable
def batchedRZ(theta, state, qubits, N_qubits):
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


@storable
def batchedRZZ(theta, state, qubits, N_qubits):
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


@storable
def batchedRXX(theta, state, qubits, N_qubits):

    for q in qubits:
        state = H(state, [q], N_qubits)
    state = batchedRZZ(theta, state, qubits, N_qubits)
    for q in qubits:
        state = H(state, [q], N_qubits)

    return state


@storable
def batchedRYY(theta, state, qubits, N_qubits):

    for q in qubits:
        state = RX(torch.tensor(np.pi / 2), state, [q], N_qubits)
    state = batchedRZZ(theta, state, qubits, N_qubits)
    for q in qubits:
        state = RX(-torch.tensor(np.pi / 2), state, [q], N_qubits)

    return state


def hamiltonian_evolution(H, state, t, qubits, N_qubits, n_steps=100):
    batch_size = len(t)
    # #permutation = [N_qubits - q - 1 for q in qubits]
    # permutation = [N_qubits - q - 1 for q in range(N_qubits) if q not in qubits]
    # permutation += [N_qubits - q - 1 for q in qubits]
    # permutation.append(N_qubits)
    # inverse_permutation = list(np.argsort(permutation))

    # new_dim = [2] * (N_qubits - len(qubits)) + [batch_size] + [2**len(qubits)]
    # state = state.permute(*permutation).reshape((2**len(qubits), -1))

    h = t.reshape((1, -1)) / n_steps
    for _ in range(N_qubits - 1):
        h = h.unsqueeze(0)

    h = h.expand_as(state)
    # h = h.expand(2**len(qubits), -1, 2**(N_qubits - len(qubits))).reshape((2**len(qubits), -1))
    for _ in range(n_steps):
        k1 = -1j * _apply_gate(state, H, qubits, N_qubits)
        k2 = -1j * _apply_gate(state + h / 2 * k1, H, qubits, N_qubits)
        k3 = -1j * _apply_gate(state + h / 2 * k2, H, qubits, N_qubits)
        k4 = -1j * _apply_gate(state + h * k3, H, qubits, N_qubits)
        # print(state.shape)
        # k1 = -1j * torch.matmul(H, state)
        # k2 = -1j * torch.matmul(H, state + h/2 * k1)
        # k3 = -1j * torch.matmul(H, state + h/2 * k2)
        # k4 = -1j * torch.matmul(H, state + h * k3)

        state += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return state  # .reshape([2]*N_qubits + [batch_size]).permute(*inverse_permutation)
