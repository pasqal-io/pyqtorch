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

import torch

torch.set_default_dtype(torch.float64)

logging_levels = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

LOG_BASE_LEVEL = os.environ.get("PYQ_LOG_LEVEL", None)
QADENCE_LOG_LEVEL = os.environ.get("QADENCE_LOG_LEVEL", None)
LOG_LEVEL: str | None | int = QADENCE_LOG_LEVEL if not LOG_BASE_LEVEL else LOG_BASE_LEVEL


if LOG_LEVEL:
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

    logger.setLevel(LOG_LEVEL)
    [
        h.setLevel(LOG_LEVEL)  # type: ignore[func-returns-value]
        for h in logger.handlers
        if h.get_name() == "console"
    ]
    logger.debug("PyQTorch logger successfully setup")


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
    inner_prod,
    is_normalized,
    overlap,
    random_state,
    uniform_state,
    zero_state,
)
