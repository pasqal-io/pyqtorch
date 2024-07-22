from __future__ import annotations

from collections import Counter
from functools import partial
from logging import getLogger

import torch
from torch import Tensor

from pyqtorch.adjoint import AdjointExpectation
from pyqtorch.analog import Observable
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.embed import Embedding
from pyqtorch.gpsr import PSRExpectation, check_support_psr
from pyqtorch.utils import DiffMode, inner_prod, sample_multinomial

logger = getLogger(__name__)


def run(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    embedding: Embedding | None = None,
) -> Tensor:
    """Sequentially apply each operation in `circuit` to an input state `state`
    given parameter values `values`, perform an optional `embedding` on `values`
    and return an output state.

    Arguments:
        circuit: A pyqtorch.QuantumCircuit instance.
        state: A torch.Tensor of shape [2, 2, ..., batch_size].
        values: A dictionary containing <'parameter_name': torch.Tensor> pairs denoting
                the current parameter values for each parameter in `circuit`.
        embedding: An optional instance of `Embedding`.

    Returns:
         A torch.Tensor of shape [2, 2, ..., batch_size].

    Example:

    ```python exec="on" source="material-block" html="1"
    from torch import rand
    from pyqtorch import QuantumCircuit, RY, random_state, run

    n_qubits = 2
    circ = QuantumCircuit(n_qubits, [RY(0, 'theta')])
    state = random_state(n_qubits)
    run(circ, state, {'theta': rand(1)})
    ```
    """
    if embedding is not None:
        values = embedding(values)
    logger.debug(f"Running circuit {circuit} on state {state} and values {values}.")
    return circuit.run(state=state, values=values, embedding=embedding)


def sample(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    n_shots: int = 1000,
    embedding: Embedding | None = None,
) -> list[Counter]:
    """Sample from `circuit` given an input state `state` given
    current parameter values `values`, perform an optional `embedding`
    on `values` and return a list Counter objects mapping from
    <bitstring: num_samples>.

    Arguments:
        circuit: A pyqtorch.QuantumCircuit instance.
        state: A torch.Tensor of shape [2, 2, ..., batch_size].
        values: A dictionary containing <'parameter_name': torch.Tensor> pairs
                denoting the current parameter values for each parameter in `circuit`.
        n_shots: A positive int denoting the number of requested samples.
        embedding: An optional instance of `Embedding`.

    Returns:
         A list of Counter objects containing <bitstring: num_samples> pairs.

    Example:

    ```python exec="on" source="material-block" html="1"
    from torch import rand
    from pyqtorch import random_state, sample, QuantumCircuit, RY

    n_qubits = 2
    circ = QuantumCircuit(n_qubits, [RY(0, 'theta')])
    state = random_state(n_qubits)
    sample(circ, state, {'theta': rand(1)}, n_shots=1000)[0]
    ```
    """
    logger.debug(
        f"Sampling circuit {circuit} on state {state} and values {values} with n_shots {n_shots}."
    )
    return circuit.sample(
        state=state, values=values, n_shots=n_shots, embedding=embedding
    )


def analytical_expectation(
    circuit: QuantumCircuit,
    state: Tensor,
    observable: Observable,
    values: dict[str, Tensor] = dict(),
    embedding: Embedding | None = None,
) -> Tensor:
    """Non sampled expectation value.

    Given an initial state :math:`\\ket\\rangle`,
    a quantum circuit :math:`U(\\theta)`,
    the analytical expectation value with :math:`O` is defined as
    :math:`\\langle\\bra U_{\\dag}(\\theta) |O| U(\\theta) \\ket\\rangle`

    Args:
        circuit (QuantumCircuit): Quantum circuit :math:`U(\\theta)`.
        state (Tensor): Input state :math:`\\ket\\rangle`.
        observable (Observable): Observable O.
        values (dict[str, Tensor], optional): Parameter values for the observable if any.
        embedding (Embedding | None, optional): An optional instance of `Embedding`.

    Returns:
        Tensor: Expectation value.
    """
    state = run(circuit, state, values, embedding=embedding)
    return inner_prod(state, run(observable, state, values, embedding=embedding)).real


def sampled_expectation(
    circuit: QuantumCircuit,
    state: Tensor,
    observable: Observable,
    values: dict[str, Tensor] = dict(),
    embedding: Embedding | None = None,
    n_shots: int = 1,
) -> Tensor:
    """Expectation value approximated via sampling.

    Given an input state :math:`\\ket\\rangle`,
    the expectation analytical value with :math:`O` is defined as
    :math:`\\langle\\bra|O\\ket\\rangle`

    Args:
        state (Tensor): Input state :math:`\\ket\\rangle`.
        observable (Observable): Observable O.
        values (dict[str, Tensor], optional): Parameter values for the observable if any.
        embedding (Embedding | None, optional): An optional instance of `Embedding`.
        n_shots: (int, optional): Number of samples to compute expectation on.

    Returns:
        Tensor: Expectation value.
    """
    state = run(circuit, state, values, embedding=embedding)
    n_qubits = len(state.shape)
    eigvals, eigvecs = torch.linalg.eig(
        observable.tensor(n_qubits=n_qubits, values=values).permute((2, 0, 1))
    )
    eigvec_state_prod = torch.multiply(eigvecs.flatten(), torch.conj(state.T).flatten())
    probs = torch.abs(torch.pow(eigvec_state_prod, 2))
    normalized_samples = sample_multinomial(
        probs, n_qubits, n_shots, normalize=True, return_counter=False
    )
    return torch.einsum(
        "i,ji ->", eigvals, normalized_samples.reshape([2] * n_qubits)  # type: ignore[union-attr]
    ).real


def expectation(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    observable: Observable = None,  # type: ignore[assignment]
    diff_mode: DiffMode = DiffMode.AD,
    options: dict[str, int] = dict(),
    embedding: Embedding | None = None,
) -> Tensor:
    """Compute the expectation value of `circuit` given a `state`,
    parameter values `values` and an `observable`
    and optionally compute gradients using diff_mode.

    Arguments:
        circuit: A pyqtorch.QuantumCircuit instance.
        state: A torch.Tensor of shape [2, 2, ..., batch_size].
        values: A dictionary containing <'parameter_name': torch.Tensor> pairs
                denoting the current parameter values for each parameter in `circuit`.
        observable: A pyq.Observable instance.
        diff_mode: The differentiation mode.
        options (dict): a dict of options infer the expectation function.
        If contains `n_shots`, expectations are computed after sampling `n_shots`.
        Only used with DiffMode.GPSR or DiffMode.AD.
        embedding: An optional instance of `Embedding`.

    Returns:
        An expectation value.

    Example:

    ```python exec="on" source="material-block" html="1"
    from torch import pi, ones_like, tensor
    from pyqtorch import random_state, RY, expectation, DiffMode, Observable, Add, Z, QuantumCircuit
    from torch.autograd import grad

    n_qubits = 2
    circ = QuantumCircuit(n_qubits, [RY(0, 'theta')])
    state = random_state(n_qubits)
    theta = tensor(pi, requires_grad=True)
    observable = Observable(n_qubits, Add([Z(i) for i in range(n_qubits)]))
    expval = expectation(circ, state, {'theta': theta}, observable, diff_mode = DiffMode.ADJOINT)
    dfdtheta= grad(expval, theta, ones_like(expval))[0]
    ```
    """

    if embedding is not None and diff_mode != DiffMode.AD:
        raise NotImplementedError("Only diff_mode AD supports embedding")
    logger.debug(
        f"Computing expectation of circuit {circuit} on state {state}, values {values},\
          given observable {observable} and diff_mode {diff_mode}."
    )
    if observable is None:
        logger.error("Please provide an observable to compute expectation.")

    if state is None:
        state = circuit.init_state(batch_size=1)

    expectation_fn = analytical_expectation
    n_shots = options.get("n_shots")
    if n_shots is not None:
        if isinstance(n_shots, int) and n_shots > 0:
            expectation_fn = partial(sampled_expectation, n_shots=n_shots)
        else:
            logger.error("Please provide a 'n_shots' in options of type 'int'.")

    if diff_mode == DiffMode.AD:
        return expectation_fn(circuit, state, observable, values, embedding)
    elif diff_mode == DiffMode.ADJOINT:
        return AdjointExpectation.apply(
            circuit,
            state,
            observable,
            embedding,
            values.keys(),
            *values.values(),
        )
    elif diff_mode == DiffMode.GPSR:
        check_support_psr(circuit)
        return PSRExpectation.apply(
            circuit, state, observable, embedding, values.keys(), *values.values()
        )
    else:
        logger.error(f"Requested diff_mode '{diff_mode}' not supported.")
