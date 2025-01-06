from __future__ import annotations

from collections import Counter
from functools import partial
from logging import getLogger

import torch
from torch import Tensor

from pyqtorch.apply import apply_operator, apply_operator_dm
from pyqtorch.circuit import QuantumCircuit
from pyqtorch.differentiation import (
    AdjointExpectation,
    PSRExpectation,
    check_support_psr,
)
from pyqtorch.embed import Embedding
from pyqtorch.hamiltonians import Observable
from pyqtorch.utils import DensityMatrix, DiffMode, sample_multinomial

logger = getLogger(__name__)


def run(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] | None = None,
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
    values = values or dict()
    logger.debug(f"Running circuit {circuit} on state {state} and values {values}.")
    return circuit.run(state=state, values=values, embedding=embedding)


def sample(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] | None = None,
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
    values = values or dict()
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
    values: dict[str, Tensor] | None = None,
    embedding: Embedding | None = None,
) -> Tensor:
    """Compute the analytical expectation value.

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
    values = values or dict()
    state = run(circuit, state, values, embedding=embedding)
    return observable.expectation(state, values, embedding=embedding)


def sampled_expectation(
    circuit: QuantumCircuit,
    state: Tensor,
    observable: Observable,
    values: dict[str, Tensor] | None = None,
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
    values = values or dict()
    state = run(circuit, state, values, embedding=embedding)
    n_qubits = circuit.n_qubits

    # batchsize needs to be first dim for eigh
    eigvals, eigvecs = torch.linalg.eigh(
        observable.tensor(
            values=values, embedding=embedding, full_support=tuple(range(n_qubits))
        ).permute((2, 0, 1))
    )
    eigvals = eigvals.squeeze()
    if isinstance(state, DensityMatrix):
        eigvec_state_prod = apply_operator_dm(
            state,
            eigvecs.T.conj(),
            tuple(range(n_qubits)),
        )
        probs = torch.diagonal(eigvec_state_prod, dim1=0, dim2=1).real

    else:
        eigvec_state_prod = apply_operator(
            state,
            eigvecs.T.conj(),
            tuple(range(n_qubits)),
        )
        eigvec_state_prod = torch.flatten(
            eigvec_state_prod, start_dim=0, end_dim=-2
        ).t()
        probs = torch.pow(torch.abs(eigvec_state_prod), 2)
    if circuit.readout_noise is not None:
        batch_samples = circuit.readout_noise.apply(probs, n_shots)

    batch_sample_multinomial = torch.func.vmap(
        lambda p: sample_multinomial(
            p, n_qubits, n_shots, return_counter=False, minlength=probs.shape[-1]
        ),
        randomness="different",
    )
    batch_samples = batch_sample_multinomial(probs)

    normalized_samples = torch.div(
        batch_samples, torch.tensor(n_shots, dtype=probs.dtype)
    )
    normalized_samples.requires_grad = True
    expectations = torch.einsum(
        "i,ji ->j", eigvals.to(dtype=normalized_samples.dtype), normalized_samples
    )
    return expectations


def expectation(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] | None = None,
    observable: Observable = None,  # type: ignore[assignment]
    diff_mode: DiffMode = DiffMode.AD,
    n_shots: int | None = None,
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
        n_shots: Number of shots for estimating expectation values.
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
    observable = Observable(Add([Z(i) for i in range(n_qubits)]))
    expval = expectation(circ, state, {'theta': theta}, observable, diff_mode = DiffMode.ADJOINT)
    dfdtheta= grad(expval, theta, ones_like(expval))[0]
    ```
    """
    values = values or dict()
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
            circuit,
            state,
            observable,
            embedding,
            expectation_fn,
            values.keys(),
            *values.values(),
        )
    else:
        logger.error(f"Requested diff_mode '{diff_mode}' not supported.")
