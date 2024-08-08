from __future__ import annotations

import sys


class NoiseProtocol:
    BITFLIP = "BitFlip"
    PHASEFLIP = "PhaseFlip"
    DEPOLARIZING = "Depolarizing"
    PAULI_CHANNEL = "PauliChannel"
    AMPLITUDE_DAMPING = "AmplitudeDamping"
    PHASE_DAMPING = "PhaseDamping"
    GENERALIZED_AMPLITUDE_DAMPING = "GeneralizedAmplitudeDamping"

    def __init__(self, protocol: str, options: dict = dict()) -> None:
        self.protocol: str = protocol
        self.options: dict = options

    def __repr__(self) -> str:
        if self.target:
            return (
                f"{self.protocol}(prob: {self.error_probability}, "
                f"target: {self.target})"
            )
        return f"{self.protocol}(prob: {self.error_probability})"

    @property
    def error_probability(self):
        return self.options.get("error_probability")

    @property
    def target(self):  #! init_state not good size
        return self.options.get("target")

    def protocol_to_gate(self):
        try:
            gate_class = getattr(sys.modules["pyqtorch.noise.gates"], self.protocol)
            return gate_class
        except AttributeError:
            raise ValueError(
                f"The protocol {self.protocol} has not been implemented in pyq yet."
            )


def _repr_noise(noise: NoiseProtocol | dict[str, NoiseProtocol] | None = None) -> str:
    """Returns the string for noise representation in gates."""
    noise_info = ""
    if noise is None:
        return noise_info
    elif isinstance(noise, NoiseProtocol):
        noise_info = str(noise)
    elif isinstance(noise, dict):
        noise_info = ", ".join(str(noise_instance) for noise_instance in noise.values())
    return f", noise: {noise_info}"
