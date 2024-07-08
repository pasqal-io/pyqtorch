from __future__ import annotations

from collections import Counter
from logging import getLogger

from torch import Tensor

from pyqtorch.adjoint import AdjointExpectation
from pyqtorch.analog import Observable
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.gpsr import PSRExpectation, check_support_psr
from pyqtorch.measurement import Measurements
from pyqtorch.utils import DiffMode, inner_prod

logger = getLogger(__name__)


def run(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
) -> Tensor:
    """Sequentially apply each operation in `circuit.operations` to an input state `state`
    given current parameter values `values`, perform an optional `embedding` on `values`
    and return an output state.

    Arguments:
    circuit: A pyqtorch.QuantumCircuit instance.
    state: A torch.Tensor of shape [2, 2, ..., batch_size].
    values: A dictionary containing `parameter_name`: torch.Tensor key,value pairs denoting
            the current parameter values for each parameter in `circuit`.
    Returns:
         A torch.Tensor of shape [2, 2, ..., batch_size]
    """
    logger.debug(f"Running circuit {circuit} on state {state} and values {values}.")
    return circuit.run(state, values)


def sample(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    n_shots: int = 1000,
) -> list[Counter]:
    """Sample from `circuit` given an input state `state` given current parameter values `values`,
       perform an optional `embedding` on `values` and return a list Counter objects mapping from
       bitstring: num_samples.

    Arguments:
    circuit: A pyqtorch.QuantumCircuit instance.
    state: A torch.Tensor of shape [2, 2, ..., batch_size].
    values: A dictionary containing `parameter_name`: torch.Tensor key,value pairs
            denoting the current parameter values for each parameter in `circuit`.
    n_shots: A positive int denoting the number of requested samples.
    Returns:
         A list of Counter objects containing bitstring:num_samples pairs.
    """
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
    measurement: Measurements | None = None
) -> Tensor:
    """Compute the expectation value of `circuit` given a `state`, parameter values `values`
        given an `observable` and optionally compute gradients using diff_mode.
    Arguments:
        circuit: A pyqtorch.QuantumCircuit instance.
        state: A torch.Tensor of shape [2, 2, ..., batch_size].
        values: A dictionary containing `parameter_name`: torch.Tensor key,value pairs
                denoting the current parameter values for each parameter in `circuit`.
        observable: A pyq.Observable instance.
        diff_mode: The differentiation mode.
    Returns:
        An expectation value.
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
            circuit, state, observable, values.keys(), *values.values()
        )
    elif diff_mode == DiffMode.GPSR:
        check_support_psr(circuit)
        return PSRExpectation.apply(
            circuit, state, observable, values.keys(), *values.values()
        )
    else:
        logger.error(f"Requested diff_mode '{diff_mode}' not supported.")
