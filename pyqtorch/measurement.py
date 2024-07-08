from utils import MeasurementMode

class Measurements:
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
        }
    
    def _tomography_expectation(self):
        n_shots = self.options.get("n_shots")
        if n_shots is None:
            raise KeyError("Tomography protocol requires a 'n_shots' kwarg of type 'int' or 'list[int]').")
        raise NotImplementedError

        # estimated_values = []
        # for observable in observables:
        #     pauli_decomposition = unroll_block_with_scaling(observable)
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