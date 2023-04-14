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

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from pyqtorch.core.circuit import QuantumCircuit
from pyqtorch.core.operation import CNOT, RX, RY, RZ, U


class OneLayerRotation(QuantumCircuit):
    def __init__(self, n_qubits: int, arbitrary: bool = False):
        super().__init__(n_qubits)
        self.theta: nn.Parameter
        if arbitrary:
            self.theta = nn.Parameter(torch.empty((self.n_qubits, 3)))
        else:
            self.theta = nn.Parameter(torch.empty((self.n_qubits,)))
        self.reset_parameters()
        self.arbitrary = arbitrary

    def reset_parameters(self) -> None:
        init.uniform_(self.theta, -2 * np.pi, 2 * np.pi)

    def forward(self, state: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.arbitrary:
            for i, t in enumerate(self.theta):
                state = U(t[0], t[1], t[2], state, [i], self.n_qubits)
        return state


class OneLayerXRotation(OneLayerRotation):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)

    def forward(self, state: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, t in enumerate(self.theta):
            state = RX(t, state, [i], self.n_qubits)
        return state


class OneLayerYRotation(OneLayerRotation):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)

    def forward(self, state: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, t in enumerate(self.theta):
            state = RY(t, state, [i], self.n_qubits)
        return state


class OneLayerZRotation(OneLayerRotation):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)

    def forward(self, state: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, t in enumerate(self.theta):
            state = RZ(t, state, [i], self.n_qubits)
        return state


class OneLayerEntanglingAnsatz(QuantumCircuit):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)
        self.param_layer = OneLayerRotation(n_qubits=self.n_qubits, arbitrary=True)

    def forward(self, state: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        state = self.param_layer(state)
        for i in range(self.n_qubits):
            state = CNOT(state, [i % self.n_qubits, (i + 1) % self.n_qubits], self.n_qubits)
        return state


class AlternateLayerAnsatz(QuantumCircuit):
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__(n_qubits)
        self.layers = nn.ModuleList(
            [OneLayerEntanglingAnsatz(self.n_qubits) for _ in range(n_layers)]
        )

    def forward(self, state: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            state = layer(state)
        return state
