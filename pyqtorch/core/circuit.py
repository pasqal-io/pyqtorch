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

class QuantumCircuit(nn.Module):

    def __init__(self, n_qubits):
        super(QuantumCircuit, self).__init__()
        self.n_qubits = n_qubits

    def init_state(self, batch_size=1, device='cpu'):
        state = torch.zeros((2**self.n_qubits, batch_size),
                            dtype=torch.cdouble).to(device)
        state[0] = 1
        state = state.reshape([2] * self.n_qubits + [batch_size])
        return state

    def uniform_state(self, batch_size=1, device='cpu'):
        state = torch.ones((2**self.n_qubits, batch_size),
                            dtype=torch.cdouble).to(device)
        state = state / torch.sqrt(torch.tensor(2**self.n_qubits))
        state = state.reshape([2] * self.n_qubits + [batch_size])
        return state
