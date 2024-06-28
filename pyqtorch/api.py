from __future__ import annotations

from collections import Counter
from logging import getLogger

from torch import Tensor, nn

from pyqtorch.adjoint import AdjointExpectation
from pyqtorch.analog import Observable
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.utils import DiffMode, inner_prod

logger = getLogger(__name__)


class Api(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


def run(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
) -> Tensor:
    """Run a circuit given a state and values dict."""
    logger.debug(f"Running circuit {circuit} on state {state} and values {values}.")
    return circuit.run(state, values)


def sample(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    n_shots: int = 1000,
) -> list[Counter]:
    """Sample a circuit given a state and values dict with n_shots."""
    logger.debug(
        f"Sampling circuit {circuit} on state {state} and values {values} with n_shots {n_shots}."
    )
    return circuit.sample(state, values, n_shots)


def expectation(
    circuit: QuantumCircuit,
    state: Tensor,
    values: dict[str, Tensor],
    observable: Observable,
    diff_mode: DiffMode = DiffMode.AD,
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
        state = circuit.run(state, values)
        return inner_prod(state, observable.run(state, values)).real
    elif diff_mode == DiffMode.ADJOINT:
        return AdjointExpectation.apply(
            circuit, observable, state, values.keys(), *values.values()
        )
    else:
        logger.error(f"Requested diff_mode '{diff_mode}' not supported.")
