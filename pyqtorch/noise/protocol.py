from __future__ import annotations

import sys

from pyqtorch.utils import StrEnum


class NoiseType(StrEnum):
    BITFLIP = "BitFlip"
    PHASEFLIP = "PhaseFlip"
    DEPOLARIZING = "Depolarizing"
    PAULI_CHANNEL = "PauliChannel"
    AMPLITUDE_DAMPING = "AmplitudeDamping"
    PHASE_DAMPING = "PhaseDamping"
    GENERALIZED_AMPLITUDE_DAMPING = "GeneralizedAmplitudeDamping"


class NoiseProtocol:
    """
    Defines a protocol of noisy quantum channels.

    Args:
        protocol: single NoiseType instance or list of NoiseType instances, or
            list of (NoiseType, options) tuple. When passing list of tuples,
            for each NoiseType the options should be a dict containing the "error_probability",
            and optionally a "target". If no "target" is present, the noise instance
            will be applied to the same target of the gate it is used on.
        error_probability: probability of error when passing a single NoiseType
            or a list of NoiseTypes. Note that all noise types require a single float for the
            error_probability, while the PauliChannel and GeneralizedAmplitudeDamping require
            a tuple of error probabilities.

    Examples:
    ```
        from pyqtorch.noise import NoiseProtocol, NoiseType

        # Single noise instance
        protocol = NoiseProtocol(NoiseType.BITFLIP, error_probability = 0.5)

        # Multiples noise instances with same probability
        protocol = NoiseProtocol(
                        [NoiseType.BITFLIP, NoiseType.PHASEFLIP],
                        error_probability = 0.5
                    )

        # Multiples noise instances with different options
        protocol = NoiseProtocol([
                        (NoiseType.BITFLIP, {"error_probability": 0.5}),
                        (NoiseType.PHASEFLIP, {"error_probability": 0.2, "target": 0}),
                        (NoiseType.PAULI_CHANNEL, {"error_probability": (0.1, 0.2, 0.2)})
                    ])

    ```
    """

    def __init__(
        self,
        protocol: NoiseType | list[NoiseType] | list[tuple[NoiseType, dict]],
        error_probability: tuple[float, ...] | float | None = None,
    ) -> None:

        if isinstance(protocol, NoiseType):
            self.protocol = [(protocol, {"error_probability": error_probability})]  # type: ignore [misc]
        elif isinstance(protocol, list) and isinstance(protocol[0], NoiseType):
            self.protocol = [
                (p, {"error_probability": error_probability}) for p in protocol  # type: ignore [misc]
            ]
        else:
            self.protocol = protocol  # type: ignore [assignment]

        for _, options in self.protocol:
            err = options.get("error_probability")
            if err is None:
                raise ValueError(
                    "Found missing error_probability in noise protocol. Pass it directly in the "
                    "error_probability argument or in the options dict for each noise instance."
                )

        self.len = len(self.protocol)

    @property
    def gates(self) -> list:
        gate_list = []
        for noise, options in self.protocol:
            try:
                gate_class = getattr(sys.modules["pyqtorch.noise.gates"], str(noise))
                gate_list.append((gate_class, options))
            except AttributeError:
                raise ValueError(
                    f"The protocol {str(noise)} has not been implemented in pyq yet."
                )
        return gate_list

    def __repr__(self) -> str:
        if self.len == 1:
            noise, options = self.protocol[0]
            error_probability = options.get("error_probability")
            target = options.get("target")
            if target is not None:
                return f"{str(noise)}(prob: {error_probability}, target: {target})"
            else:
                return f"{str(noise)}(prob: {error_probability})"
        else:
            return f"NoiseProtocol(length = {self.len})"


def _repr_noise(noise: NoiseProtocol | None = None) -> str:
    """Returns the string for noise representation in gates."""
    noise_info = ""
    if noise is None:
        return noise_info
    elif isinstance(noise, NoiseProtocol):
        noise_info = str(noise)
    return f", noise: {noise_info}"
