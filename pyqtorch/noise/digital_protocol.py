from __future__ import annotations

import sys
from dataclasses import dataclass

from pyqtorch.utils import StrEnum


class DigitalNoiseType(StrEnum):
    BITFLIP = "BitFlip"
    PHASEFLIP = "PhaseFlip"
    DEPOLARIZING = "Depolarizing"
    PAULI_CHANNEL = "PauliChannel"
    AMPLITUDE_DAMPING = "AmplitudeDamping"
    PHASE_DAMPING = "PhaseDamping"
    GENERALIZED_AMPLITUDE_DAMPING = "GeneralizedAmplitudeDamping"


@dataclass
class DigitalNoiseInstance:
    type: DigitalNoiseType
    error_probability: tuple[float, ...] | float | None
    target: int | None


class DigitalNoiseProtocol:
    """
    Define a protocol of noisy quantum channels.

    Args:
        protocol: single DigitalNoiseType instance or list of DigitalNoiseType instances, or
            list of (DigitalNoiseType, options) tuple. When passing list of tuples, for
            each DigitalNoiseType the options should be a dict containing the "error_probability",
            and optionally a "target". If no "target" is present, the noise instance
            will be applied to the same target of the gate it is used on.
        error_probability: probability of error when passing a single DigitalNoiseType
            or a list of NoiseTypes. Note that all noise types require a single float for the
            error_probability, while the PauliChannel and GeneralizedAmplitudeDamping require
            a tuple of error probabilities.

    Examples:
    ```
        from pyqtorch.noise import DigitalNoiseProtocol, DigitalNoiseType

        # Single noise instance
        protocol = DigitalNoiseProtocol(DigitalNoiseType.BITFLIP, error_probability = 0.5)

        # Equivalent to using the respective class method
        protocol = DigitalNoiseProtocol.bitflip(error_probability = 0.5)

        # Multiples noise instances with same probability
        protocol = DigitalNoiseProtocol(
                        [DigitalNoiseType.BITFLIP, DigitalNoiseType.PHASEFLIP],
                        error_probability = 0.5
                    )

        # Multiples noise instances with different options
        prot0 = DigitalNoiseProtocol.bitflip(0.5)
        prot1 = DigitalNoiseProtocol.pauli_channel((0.1, 0.2, 0.7))
        protocol = DigitalNoiseProtocol([prot0, prot1])

    ```
    """

    def __init__(
        self,
        protocol: (
            DigitalNoiseType | list[DigitalNoiseType] | list[DigitalNoiseProtocol]
        ),
        error_probability: tuple[float, ...] | float | None = None,
        target: int | None = None,
    ) -> None:

        self._error_probability = error_probability
        self._target = target

        is_protocol_list = isinstance(protocol, list)

        if isinstance(protocol, DigitalNoiseType):
            self.noise_instances = [
                DigitalNoiseInstance(protocol, error_probability, target)
            ]
        elif is_protocol_list and isinstance(protocol[0], DigitalNoiseType):
            self.noise_instances = [
                DigitalNoiseInstance(p, error_probability, target) for p in protocol  # type: ignore [arg-type]
            ]
        elif is_protocol_list and isinstance(protocol[0], DigitalNoiseProtocol):
            self.noise_instances = []
            for p in protocol:
                self.noise_instances.extend(p.noise_instances)  # type: ignore [union-attr]
        else:
            raise TypeError(f"Incorrect protocol type: {type(protocol)}.")

        for noise in self.noise_instances:
            if noise.error_probability is None:
                raise ValueError(
                    f"No error_probability passed to the protocol {noise.type}."
                )

        self.len = len(self.noise_instances)

    @property
    def error_probability(self):
        if self.len == 1:
            return self._error_probability
        else:
            return [noise.error_probability for noise in self.noise_instances]

    @property
    def target(self):
        if self.len == 1:
            return self._target
        else:
            return [noise.target for noise in self.noise_instances]

    @property
    def gates(self) -> list:
        gate_list = []
        for noise in self.noise_instances:
            try:
                gate_class = getattr(
                    sys.modules["pyqtorch.noise.digital_gates"], str(noise.type)
                )
                gate_list.append((gate_class, noise))
            except AttributeError:
                raise ValueError(
                    f"The protocol {str(noise.type)} has not been implemented in PyQTorch yet."
                )
        return gate_list

    @classmethod
    def bitflip(cls, *args, **kwargs) -> DigitalNoiseProtocol:
        return cls(DigitalNoiseType.BITFLIP, *args, **kwargs)

    @classmethod
    def phaseflip(cls, *args, **kwargs) -> DigitalNoiseProtocol:
        return cls(DigitalNoiseType.PHASEFLIP, *args, **kwargs)

    @classmethod
    def depolarizing(cls, *args, **kwargs) -> DigitalNoiseProtocol:
        return cls(DigitalNoiseType.DEPOLARIZING, *args, **kwargs)

    @classmethod
    def pauli_channel(cls, *args, **kwargs) -> DigitalNoiseProtocol:
        return cls(DigitalNoiseType.PAULI_CHANNEL, *args, **kwargs)

    @classmethod
    def amplitude_damping(cls, *args, **kwargs) -> DigitalNoiseProtocol:
        return cls(DigitalNoiseType.AMPLITUDE_DAMPING, *args, **kwargs)

    @classmethod
    def phase_damping(cls, *args, **kwargs) -> DigitalNoiseProtocol:
        return cls(DigitalNoiseType.PHASE_DAMPING, *args, **kwargs)

    @classmethod
    def generalized_amplitude_damping(cls, *args, **kwargs) -> DigitalNoiseProtocol:
        return cls(DigitalNoiseType.GENERALIZED_AMPLITUDE_DAMPING, *args, **kwargs)

    def __repr__(self) -> str:
        if self.len == 1:
            noise = self.noise_instances[0]
            if noise.target is not None:
                return f"{str(noise.type)}(prob: {noise.error_probability}, target: {noise.target})"
            else:
                return f"{str(noise.type)}(prob: {noise.error_probability})"
        else:
            return f"DigitalNoiseProtocol(length = {self.len})"


def _repr_noise(noise: DigitalNoiseProtocol | None = None) -> str:
    """Returns the string for noise representation in gates."""
    noise_info = ""
    if noise is None:
        return noise_info
    elif isinstance(noise, DigitalNoiseProtocol):
        noise_info = str(noise)
    return f", noise: {noise_info}"
