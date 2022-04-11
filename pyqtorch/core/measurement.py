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

from cgi import parse_multipart
import torch
from pyqtorch.core.operation import X, Y, Z

qubit_operators = {'X': X, 'Y': Y, 'Z': Z}


def total_magnetization(state, N_qubits, batch_size):
    new_state = torch.zeros_like(state)

    for i in range(N_qubits):
        new_state += Z(state, [i], N_qubits)

    state = state.reshape((2**N_qubits, batch_size))
    new_state = new_state.reshape((2**N_qubits, batch_size))

    return torch.real(
        torch.sum(torch.conj(state) * new_state, axis=0))


def measure_openfermion(state, operator, N_qubits, batch_size):
    new_state = torch.zeros_like(state)

    for op, coef in operator.terms.items():
        for qubit, pauli in op:
            state_bis = qubit_operators[pauli](state, [qubit], N_qubits)
            new_state += state_bis * coef

    state = state.reshape((2**N_qubits, batch_size))
    new_state = new_state.reshape((2**N_qubits, batch_size))

    return torch.real(
        torch.sum(torch.conj(state) * new_state, axis=0))
