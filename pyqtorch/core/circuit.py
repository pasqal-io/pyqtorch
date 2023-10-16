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
from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE


class QuantumCircuit(nn.Module):
    def __init__(self, n_qubits: int):
        super(QuantumCircuit, self).__init__()
        self.n_qubits = n_qubits

    def init_state(
        self, batch_size: int = 1, device: Union[str, torch.device] = "cpu"
    ) -> torch.Tensor:
        state = torch.zeros(
            (2**self.n_qubits, batch_size), dtype=DEFAULT_MATRIX_DTYPE, device=device
        )
        state[0] = 1
        state = state.reshape([2] * self.n_qubits + [batch_size])
        return state

    def uniform_state(
        self, batch_size: int = 1, device: Union[str, torch.device] = "cpu"
    ) -> torch.Tensor:
        state = torch.ones(
            (2**self.n_qubits, batch_size), dtype=DEFAULT_MATRIX_DTYPE, device=device
        )
        state = state / torch.sqrt(torch.tensor(2**self.n_qubits))
        state = state.reshape([2] * self.n_qubits + [batch_size])
        return state
