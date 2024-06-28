from __future__ import annotations

from collections import Counter
from logging import getLogger
from typing import Any

from torch import Tensor, nn

from pyqtorch.adjoint import AdjointExpectation
from pyqtorch.analog import Observable
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.embed import Embedding
from pyqtorch.utils import DiffMode, inner_prod

logger = getLogger(__name__)


class Api(nn.Module):
    def __init__(
        self,
        circuit: QuantumCircuit,
        observable: Observable,
        embedding: Embedding,
        args: Any,
        kwargs: Any,
    ) -> None:
        super().__init__()
        self.circuit = circuit
        self.observable = observable
        self.embedding = embedding

    def run(self) -> Tensor:
        pass


def run(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    embedding: Embedding | None = None,
) -> Tensor:
    """Run a circuit given a state and values dict."""
    logger.debug(f"Running circuit {circuit} on state {state} and values {values}.")
    return circuit.run(state, values, embedding)


def sample(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    n_shots: int = 1000,
    embedding: Embedding | None = None,
) -> list[Counter]:
    """Sample a circuit given a state and values dict with n_shots."""
    logger.debug(
        f"Sampling circuit {circuit} on state {state} and values {values} with n_shots {n_shots}."
    )
    return circuit.sample(state, values, n_shots, embedding)


def expectation(
    circuit: QuantumCircuit,
    state: Tensor,
    values: dict[str, Tensor],
    observable: Observable,
    diff_mode: DiffMode = DiffMode.AD,
    embedding: Embedding | None = None,
) -> Tensor:
    """Compute the expectation value of the circuit given a state, values dict, observable
       and optionally compute gradients using diff_mode.
    Arguments:
        circuit: QuantumCircuit instance
        state: An input state
        values: A dictionary of parameter values
        observable: Hamiltonian representing the observable
        diff_mode: The differentiation mode
    Returns:
        A expectation value.
    """
    logger.debug(
        f"Computing expectation of circuit {circuit} on state {state}, values {values},\
          given observable {observable} and diff_mode {diff_mode}."
    )
    if observable is None:
        logger.error("Please provide an observable to compute expectation.")
    if state is None:
        state = circuit.init_state(batch_size=1)
    if diff_mode == DiffMode.AD:
        state = circuit.run(state, values, embedding)
        return inner_prod(state, observable.run(state, values, embedding)).real
    elif diff_mode == DiffMode.ADJOINT:
        if embedding is not None:
            logger.error("Adjoint is not supported with Embedding-Mode.")
        return AdjointExpectation.apply(
            circuit, observable, state, values.keys(), *values.values()
        )
    else:
        logger.error(f"Requested diff_mode '{diff_mode}' not supported.")
