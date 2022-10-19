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

import torch
import torch.nn as nn
from torch import Tensor

from pyqtorch.converters.store_ops import ops_cache


class QuantumCircuit(nn.Module):
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits

        # cleanup the circuit visualizer everytime a forward pass is called
        self.register_forward_pre_hook(ops_cache.clear)

    def init_state(self, batch_size: int = 1, device: str = "cpu") -> Tensor:
        state = torch.zeros((2**self.n_qubits, batch_size), dtype=torch.cdouble).to(
            device
        )
        state[0] = 1
        state = state.reshape([2] * self.n_qubits + [batch_size])
        return state

    def uniform_state(self, batch_size: str = 1, device: str = "cpu") -> Tensor:
        state = torch.ones((2**self.n_qubits, batch_size), dtype=torch.cdouble).to(
            device
        )
        state = state / torch.sqrt(torch.tensor(2**self.n_qubits))
        state = state.reshape([2] * self.n_qubits + [batch_size])
        return state
