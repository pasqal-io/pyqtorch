from __future__ import annotations

from typing import Callable

from torch import Tensor
from pyqtorch.utils import MeasurementMode


def iterate_pauli_decomposition(
    circuit,
    param_values,
    pauli_decomposition,
    n_shots: int,
    state,
) -> Tensor:
    """Estimate total expectation value by averaging all Pauli terms.

    Args:
        circuit: The circuit that is executed.
        param_values: Parameters of the circuit.
        pauli_decomposition: A list of Pauli decomposed terms.
        n_shots: Number of shots to sample.
        state: Initial state.

    Returns: A torch.Tensor of bit strings n_shots x n_qubits.
    """


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
                "Tomography protocol requires a 'n_shots' kwarg of type 'int' or 'list[int]')."
            )
        raise NotImplementedError

        # def

        # estimated_values = []
        # for observable_term in observable.operations:
        #     estimated_values.append(
        #         iterate_pauli_decomposition(
        #             circuit=circuit,
        #             param_values=param_values,
        #             pauli_decomposition=pauli_decomposition,
        #             n_shots=n_shots,
        #             state=state,
        #             backend=backend,
        #             noise=noise,
        #             endianness=endianness,
        #         )
        #     )
        # return torch.transpose(torch.vstack(estimated_values), 1, 0)
