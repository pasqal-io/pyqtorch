from __future__ import annotations

from collections import Counter
from functools import reduce
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from pyqtorch.analog import Observable
from pyqtorch.api import sample
from pyqtorch.circuit import Merge, QuantumCircuit
from pyqtorch.primitive import H, Primitive, SDagger, X, Y, Z
from pyqtorch.utils import MeasurementMode


def get_counts(samples: list[Counter], support: list[int]) -> list[Counter]:
    """Marginalise the probability mass function to the support.

    Args:
        samples: List of samples against which expectation value is to be computed.
        support: A list of integers representing qubit indices.

    Returns: A List[Counter] of bit strings.
    """
    return [
        reduce(
            lambda x, y: x + y,
            [
                Counter({"".join([k[i] for i in support]): sample[k]})
                for k, v in sample.items()
            ],
        )
        for sample in samples
    ]


def empirical_average(samples: list[Counter], support: list[int]) -> Tensor:
    """Compute the empirical average.

    Args:
        samples: List of samples against which expectation value is to be computed.
        support: A list of integers representing qubit indices.

    Returns: A torch.Tensor of the empirical average.
    """
    PARITY = -1
    counters = get_counts(samples, support)
    n_shots = sum(list(counters[0].values()))
    expectations = []
    for counter in counters:
        counter_exps = []
        for bitstring, count in counter.items():
            counter_exps.append(
                count * PARITY ** (sum([int(bit) for bit in bitstring]))
            )
        expectations.append(sum(counter_exps) / n_shots)
    return torch.tensor(expectations)


def get_qubit_indices_for_op(
    pauli_term: Module, op: Primitive | None = None
) -> list[int]:
    """Get qubit indices for the given op in the Pauli term if any.

    Args:
        pauli_term: Tuple of a Pauli block and a parameter.
        op: Tuple of Primitive blocks or None.

    Returns: A list of integers representing qubit indices.
    """
    blocks = getattr(pauli_term[0], "blocks", None)
    blocks = blocks if blocks is not None else [pauli_term[0]]
    indices = [
        block.qubit_support[0]
        for block in blocks
        if (op is None) or (isinstance(block, type(op)))
    ]
    return indices


def rotate(circuit: QuantumCircuit, pauli_term: Module):
    rotations = []

    for op, gate in [(X, Z), (Y, SDagger)]:
        qubit_indices = get_qubit_indices_for_op(pauli_term, op=op)
        for index in qubit_indices:
            rotations.append(gate(index) * H(index))
    return Merge(circuit.operations + rotations)


def evaluate_single_term(
    circuit: QuantumCircuit,
    param_values: dict[str, Tensor],
    pauli_term: Module,
    n_shots: int,
    state: Tensor,
) -> Tensor:
    """Estimate total expectation value by averaging all Pauli terms.

    Args:
        circuit: The circuit that is executed.
        param_values: Parameters of the circuit.
        observable_term: A list of Pauli decomposed terms.
        n_shots: Number of shots to sample.
        state: Initial state.

    Returns: A torch.Tensor of bit strings n_shots x n_qubits.
    """

    support = pauli_term.qubit_support
    rotated_circuit = rotate(circuit=circuit, pauli_term=pauli_term)

    samples = sample(rotated_circuit, state, param_values, n_shots)
    estim_values = empirical_average(samples=samples, support=support)
    return estim_values


class MeasurementProtocols:
    """Class handling measurement protocols.

    Attributes:
        protocol: Measurement protocol applied.
        options: Dictionary of options.
    """

    def __init__(self, protocol: MeasurementMode, options: dict) -> None:
        self.protocol = protocol
        self.options = options

        self._generator_map: dict = {
            MeasurementMode.TOMOGRAPHY: self._tomography_expectation,
            MeasurementMode.SHADOW: self._shadow_expectation,
        }

    def get_expectation_fn(self) -> Callable:
        return self._generator_map[self.protocol]

    def _shadow_expectation(self) -> Callable:
        raise NotImplementedError("SHADOW protocol not yet supported.")

    def _tomography_expectation(self) -> Callable:
        n_shots = self.options.get("n_shots")
        if n_shots is None:
            raise KeyError(
                "Tomography protocol requires a 'n_shots' kwarg of type 'int')."
            )
        # raise NotImplementedError

        def expectation_fn(
            circuit: QuantumCircuit,
            state: Tensor,
            observable: Observable,
            param_values: dict[str, Tensor],
            n_shots: int,
        ) -> Tensor:
            res = torch.sum(
                torch.stack(
                    [
                        evaluate_single_term(
                            circuit, param_values, term, n_shots, state
                        )
                        for term in observable.operations
                    ]
                ),
                axis=0,
            )
            # Allow for automatic differentiation.
            res.requires_grad = True
            return res

        return expectation_fn
