# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #     http://www.apache.org/licenses/LICENSE-2.0
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
from __future__ import annotations

import torch

from .analog import HamiltonianEvolution
from .apply import apply_operator
from .circuit import QuantumCircuit, expectation
from .parametric import CPHASE, CRX, CRY, CRZ, PHASE, RX, RY, RZ, U
from .primitive import (
    CNOT,
    CSWAP,
    CY,
    CZ,
    SWAP,
    H,
    I,
    N,
    Projector,
    S,
    SDagger,
    T,
    Toffoli,
    X,
    Y,
    Z,
)
from .utils import (
    is_normalized,
    overlap,
    random_state,
    uniform_state,
    zero_state,
)

torch.set_default_dtype(torch.float64)
