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

import torch

from pyqtorch.core.batched_operation import batchedRX
from pyqtorch.core.circuit import QuantumCircuit


class SingleLayerEncoding(QuantumCircuit):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)

    def forward(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_qubits):
            state = batchedRX(x, state, [i], self.n_qubits)
        return state
