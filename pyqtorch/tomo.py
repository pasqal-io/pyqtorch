from __future__ import annotations

from torch import Tensor

from pyqtorch.analog import Observable
from pyqtorch.api import run
from pyqtorch.apply import apply_operator
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.embed import Embedding
from pyqtorch.utils import DiffMode, sample_from_state


def tomography(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    observable: Observable = None,
    diff_mode: DiffMode = DiffMode.AD,
    embedding: Embedding | None = None,
    n_shots: int = 100,
) -> Tensor:

    state = run(circuit, state, values, embedding)
    for term in observable.operations:
        # rotate logic
        mu = apply_operator(state, term, term.qubit_support)
        samples = sample_from_state(mu, n_shots)
        # do stuff with samples
