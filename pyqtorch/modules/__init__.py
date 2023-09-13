from __future__ import annotations

from pyqtorch.modules.circuit import (
    EntanglingLayer,
    FeaturemapLayer,
    QuantumCircuit,
    VariationalLayer,
)
from pyqtorch.modules.hamevo import HamEvo, HamEvoEig, HamEvoExp, HamEvoType, HamiltonianEvolution
from pyqtorch.modules.parametric import CPHASE, CRX, CRY, CRZ, PHASE, RX, RY, RZ, U
from pyqtorch.modules.primitive import (
    CNOT,
    CSWAP,
    CY,
    CZ,
    SWAP,
    H,
    I,
    N,
    S,
    SDagger,
    T,
    Toffoli,
    X,
    Y,
    Z,
)
from pyqtorch.modules.utils import (
    _apply_parallel,
    flatten_wf,
    invert_endianness,
    is_normalized,
    normalize,
    overlap,
    random_state,
    uniform_state,
    zero_state,
)
