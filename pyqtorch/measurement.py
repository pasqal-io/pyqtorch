from __future__ import annotations

from typing import Callable

from pyqtorch.utils import MeasurementMode

# def rotate(circuit: QuantumCircuit, pauli_term: Module):
#     rotations = []

#     #for op, gate in [(X(0), Z), (Y(0), SDagger)]:

#     return circuit


# def evaluate_single_term(
#     circuit: QuantumCircuit,
#     param_values: dict[str, Tensor],
#     observable_term: Module,
#     n_shots: int,
#     state: Tensor,
# ) -> Tensor:
#     """Estimate total expectation value by averaging all Pauli terms.

#     Args:
#         circuit: The circuit that is executed.
#         param_values: Parameters of the circuit.
#         pauli_decomposition: A list of Pauli decomposed terms.
#         n_shots: Number of shots to sample.
#         state: Initial state.

#     Returns: A torch.Tensor of bit strings n_shots x n_qubits.
#     """

#     # TODO: do pauli term conversion here
#     # assumed this is already given
#     pauli_term = observable_term
#     support = pauli_term.qubit_support
#     rotated_circuit = rotate(circuit=circuit, pauli_term=pauli_term)


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
        raise NotImplementedError

        # def expectation_fn(
        #     circuit: QuantumCircuit,
        #     state: Tensor,
        #     observable: Observable,
        #     param_values: dict[str, Tensor],
        #     n_shots: int,
        # ) -> Tensor:
        #     return torch.sum(
        #         [
        #             evaluate_single_term(circuit, param_values, term, n_shots, state)
        #             for term in observable.operations
        #         ]
        #     )

        # return expectation_fn
