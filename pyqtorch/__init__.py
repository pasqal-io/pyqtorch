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

import logging
import os
import sys

logging_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

LOG_LEVEL: str = os.environ.get("PYQ_LOG_LEVEL", "").upper()


# if LOG_LEVEL:
LOG_LEVEL: int = logging_levels.get(LOG_LEVEL, logging.INFO)  # type: ignore[arg-type, no-redef]
# If logger not setup, add handler to stderr
# else use setup presumably from Qadence
handle = None
if __name__ not in logging.Logger.manager.loggerDict.keys():
    handle = logging.StreamHandler(sys.stderr)
    handle.set_name("console")

logger = logging.getLogger(__name__)
if handle:
    logger.addHandler(handle)
[
    h.setLevel(LOG_LEVEL)  # type: ignore[func-returns-value]
    for h in logger.handlers
    if h.get_name() == "console"
]
logger.setLevel(LOG_LEVEL)

logger.info(f"PyQTorch logger successfully setup with log level {LOG_LEVEL}")


from .adjoint import expectation
from .analog import (
    Add,
    DiagonalObservable,
    HamiltonianEvolution,
    Observable,
    Scale,
)
from .apply import apply_operator
from .circuit import Merge, QuantumCircuit, Sequence
from .noise import (
    AmplitudeDamping,
    BitFlip,
    Depolarizing,
    GeneralizedAmplitudeDamping,
    PauliChannel,
    PhaseDamping,
    PhaseFlip,
)
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
    DEFAULT_MATRIX_DTYPE,
    DEFAULT_REAL_DTYPE,
    DiffMode,
    inner_prod,
    is_normalized,
    overlap,
    product_state,
    random_state,
    uniform_state,
    zero_state,
)

__all__ = [
    "expectation",
    "Add",
    "HamiltonianEvolution",
    "Scale",
    "Merge",
    "QuantumCircuit",
    "Sequence",
    "CPHASE",
    "CRX",
    "CRY",
    "CRZ",
    "PHASE",
    "RX",
    "RY",
    "RZ",
    "U",
    "DEFAULT_MATRIX_DTYPE",
    "DEFAULT_REAL_DTYPE",
    "DiffMode",
    "inner_prod",
    "is_normalized",
    "overlap",
    "random_state",
    "uniform_state",
    "zero_state",
    "CNOT",
    "CSWAP",
    "CY",
    "CZ",
    "SWAP",
    "H",
    "I",
    "N",
    "Projector",
    "S",
    "SDagger",
    "T",
    "Toffoli",
    "X",
    "Y",
    "Z",
    "apply_operator",
    "Observable",
    "DiagonalObservable",
]
