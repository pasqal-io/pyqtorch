from __future__ import annotations

from collections import Counter
from logging import getLogger

from torch import Tensor

from pyqtorch.adjoint import AdjointExpectation
from pyqtorch.analog import Observable
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.embed import Embedding
from pyqtorch.utils import DiffMode, inner_prod

logger = getLogger(__name__)


def run(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    embedding: Embedding | None = None,
) -> Tensor:
    """Sequentially apply each operation in `circuit.operations` to an input state `state`
    given current parameter values `values`, perform an optional `embedding` on `values`
    and return an output state.

    Arguments:
    circuit: A pyqtorch.QuantumCircuit instance.
    state: A torch.Tensor of shape [2, 2, ..., batch_size].
    values: A dictionary containing `parameter_name`: torch.Tensor key,value pairs denoting
            the current parameter values for each parameter in `circuit`.
    embedding: An optional instance of `pyqtorch.Embedding` which holds
               a set of `parameter_name`: function pairs,
               associating each circuit parameter which a Callable.
    Returns:
         A torch.Tensor of shape [2, 2, ..., batch_size]
    """
    logger.debug(f"Running circuit {circuit} on state {state} and values {values}.")
    return circuit.run(state, values, embedding)


def sample(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    n_shots: int = 1000,
    embedding: Embedding | None = None,
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
    embedding: An optional instance of `pyqtorch.Embedding` which holds a set of
               `parameter_name`: function pairs, associating each circuit parameter with a Callable.
    Returns:
         A list of Counter objects containing bitstring:num_samples pairs.
    """
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
    """Compute the expectation value of `circuit` given a `state`, parameter values `values`
        given an `observable` and optionally compute gradients using diff_mode.
    Arguments:
        circuit: A pyqtorch.QuantumCircuit instance.
        state: A torch.Tensor of shape [2, 2, ..., batch_size].
        values: A dictionary containing `parameter_name`: torch.Tensor key,value pairs
                denoting the current parameter values for each parameter in `circuit`.
        embedding: An optional instance of `pyqtorch.Embedding` which holds a set of
                  `parameter_name`: fn pairs, associating each circuit parameter with a Callable.
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
        state = circuit.run(state, values, embedding)
        return inner_prod(state, observable.run(state, values, embedding)).real
    elif diff_mode == DiffMode.ADJOINT:
        if embedding is not None:
            logger.error("Adjoint is not supported with Embedding-Mode.")
        return AdjointExpectation.apply(
            circuit, observable, state, embedding, values.keys(), *values.values()
        )
    elif diff_mode == DiffMode.GPSR:
        # TODO for charles
        raise NotImplementedError("Will be implemented soon.")
    else:
        logger.error(f"Requested diff_mode '{diff_mode}' not supported.")