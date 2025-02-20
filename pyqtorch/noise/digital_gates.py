from __future__ import annotations

from math import sqrt
from typing import Any

import torch
from torch import Tensor

from pyqtorch.apply import apply_operator_dm
from pyqtorch.embed import Embedding
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, IMAT, XMAT, YMAT, ZMAT
from pyqtorch.utils import (
    DensityMatrix,
    density_mat,
    promote_operator,
    qubit_support_as_tuple,
)


class Noise(torch.nn.Module):
    def __init__(
        self,
        kraus: list[Tensor],
        target: int,
        error_probability: tuple[float, ...] | float,
    ) -> None:
        super().__init__()
        self.target: int = target
        self.qubit_support: tuple[int, ...] = qubit_support_as_tuple(self.target)
        self.is_diagonal = False
        for index, tensor in enumerate(kraus):
            self.register_buffer(f"kraus_{index}", tensor)
        self._device: torch.device = kraus[0].device
        self._dtype: torch.dtype = kraus[0].dtype
        self.error_probability: tuple[float, ...] | float = error_probability

    def extra_repr(self) -> str:
        return f"target: {self.qubit_support}, prob: {self.error_probability}"

    @property
    def kraus_operators(self) -> list[Tensor]:
        return [getattr(self, f"kraus_{i}") for i in range(len(self._buffers))]

    def tensor(self, n_qubit_support: int | None = None) -> list[Tensor]:
        # Since PyQ expects tensor.Size = [2**n_qubits, 2**n_qubits,batch_size].
        t_ops = [kraus_op.unsqueeze(2) for kraus_op in self.kraus_operators]
        if n_qubit_support is None:
            return t_ops
        return [promote_operator(t, self.target, n_qubit_support) for t in t_ops]

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | Tensor | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        """
        Applies a noisy quantum channel on the input state.
        The evolution is represented as a sum of Kraus operators:
        .. math::
            S(\\rho) = \\sum_i K_i \\rho K_i^\\dagger,

        Each Kraus operator in the `kraus_list` is applied to the input state, and the result
        is accumulated to obtain the evolved state.

        Args:
            state (Tensor): Input quantum state represented as a tensor.

        Returns:
            Tensor: Quantum state as a density matrix after evolution.

        Raises:
            TypeError: If the input `state` or `kraus_list` is not a Tensor.
        """
        values = values or dict()
        if not isinstance(state, DensityMatrix):
            state = density_mat(state)
        rho_evols: list[Tensor] = []
        for kraus in self.tensor():
            rho_evol: Tensor = apply_operator_dm(state, kraus, self.qubit_support)
            rho_evols.append(rho_evol)
        rho_final: Tensor = torch.stack(rho_evols, dim=0)
        rho_final = torch.sum(rho_final, dim=0)
        return rho_final

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, *args: Any, **kwargs: Any) -> Noise:
        super().to(*args, **kwargs)
        self._device = self.kraus_0.device
        self._dtype = self.kraus_0.dtype
        return self


class BitFlip(Noise):
    """
    Initialize the BitFlip gate.

    The bit flip channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho + p X \\rho X^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The probability of a bit flip error.

    Raises:
        TypeError: If the error_probability value is not a float.
    """

    def __init__(
        self,
        target: int,
        error_probability: float,
    ):
        if error_probability > 1.0 or error_probability < 0.0:
            raise ValueError("The error_probability value is not a correct probability")
        K0: Tensor = sqrt(1.0 - error_probability) * IMAT
        K1: Tensor = sqrt(error_probability) * XMAT
        kraus_bitflip: list[Tensor] = [K0, K1]
        super().__init__(kraus_bitflip, target, error_probability)


class PhaseFlip(Noise):
    """
    Initialize the PhaseFlip gate

    The phase flip channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho + p Z \\rho Z^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The probability of phase flip error.

    Raises:
        TypeError: If the error_probability value is not a float.
    """

    def __init__(
        self,
        target: int,
        error_probability: float,
    ):
        if error_probability > 1.0 or error_probability < 0.0:
            raise ValueError("The error_probability value is not a correct probability")
        K0: Tensor = sqrt(1.0 - error_probability) * IMAT
        K1: Tensor = sqrt(error_probability) * ZMAT
        kraus_phaseflip: list[Tensor] = [K0, K1]
        super().__init__(kraus_phaseflip, target, error_probability)


class Depolarizing(Noise):
    """
    Initialize the Depolarizing gate.

    The depolarizing channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho
            + p/3 X \\rho X^{\\dagger}
            + p/3 Y \\rho Y^{\\dagger}
            + p/3 Z \\rho Z^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The probability of depolarizing error.

    Raises:
        TypeError: If the error_probability value is not a float.
    """

    def __init__(
        self,
        target: int,
        error_probability: float,
    ):
        if error_probability > 1.0 or error_probability < 0.0:
            raise ValueError("The error_probability value is not a correct probability")
        K0: Tensor = sqrt(1.0 - error_probability) * IMAT
        K1: Tensor = sqrt(error_probability / 3) * XMAT
        K2: Tensor = sqrt(error_probability / 3) * YMAT
        K3: Tensor = sqrt(error_probability / 3) * ZMAT
        kraus_depolarizing: list[Tensor] = [K0, K1, K2, K3]
        super().__init__(kraus_depolarizing, target, error_probability)


class PauliChannel(Noise):
    """
    Initialize the PauliChannel gate.

    The pauli channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-px-py-pz) \\rho
            + px X \\rho X^{\\dagger}
            + py Y \\rho Y^{\\dagger}
            + pz Z \\rho Z^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (Tuple[float, ...]): Tuple containing probabilities
            of X, Y, and Z errors.

    Raises:
        TypeError: If the probabilities values are not provided in a tuple.
    """

    def __init__(
        self,
        target: int,
        error_probability: tuple[float, ...],
    ):
        sum_prob = sum(error_probability)
        if sum_prob > 1.0:
            raise ValueError("The sum of probabilities can't be greater than 1.0")
        for probability in error_probability:
            if probability > 1.0 or probability < 0.0:
                raise ValueError("The probability values are not correct probabilities")
        px, py, pz = (
            error_probability[0],
            error_probability[1],
            error_probability[2],
        )
        K0: Tensor = sqrt(1.0 - (px + py + pz)) * IMAT
        K1: Tensor = sqrt(px) * XMAT
        K2: Tensor = sqrt(py) * YMAT
        K3: Tensor = sqrt(pz) * ZMAT
        kraus_pauli_channel: list[Tensor] = [K0, K1, K2, K3]
        super().__init__(kraus_pauli_channel, target, error_probability)


class AmplitudeDamping(Noise):
    """
    Initialize the AmplitudeDamping gate.

    The amplitude damping channel is defined as:

    .. math::
        \\rho \\Rightarrow K_0 \\rho K_0^{\\dagger} + K_1 \\rho K_1^{\\dagger}

    with:

    .. code-block:: python

        K0 = [[1, 0], [0, sqrt(1 - rate)]]
        K1 = [[0, sqrt(rate)], [0, 0]]

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The damping rate, indicating the probability of amplitude loss.

    Raises:
        TypeError: If the rate value is not a float.
        ValueError: If the damping rate is not a correct probability.
    """

    def __init__(
        self,
        target: int,
        error_probability: float,
    ):
        rate = error_probability
        if rate > 1.0 or rate < 0.0:
            raise ValueError("The damping rate is not a correct probability")
        K0: Tensor = torch.tensor(
            [[1, 0], [0, sqrt(1 - rate)]], dtype=DEFAULT_MATRIX_DTYPE
        )
        K1: Tensor = torch.tensor([[0, sqrt(rate)], [0, 0]], dtype=DEFAULT_MATRIX_DTYPE)
        kraus_amplitude_damping: list[Tensor] = [K0, K1]
        super().__init__(kraus_amplitude_damping, target, rate)


class PhaseDamping(Noise):
    """
    Initialize the PhaseDamping gate.

    The phase damping channel is defined as:

    .. math::
        \\rho \\Rightarrow K_0 \\rho K_0^{\\dagger} + K_1 \\rho K_1^{\\dagger}

     with:

    .. code-block:: python

        K0 = [[1, 0], [0, sqrt(1 - rate)]]
        K1 = [[0, 0], [0, sqrt(rate)]]

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probability (float): The damping rate, indicating the probability of phase damping.

    Raises:
        TypeError: If the rate value is not a float.
        ValueError: If the damping rate is not a correct probability.
    """

    def __init__(
        self,
        target: int,
        error_probability: float,
    ):
        rate = error_probability
        if rate > 1.0 or rate < 0.0:
            raise ValueError("The damping rate is not a correct probability")
        K0: Tensor = torch.tensor(
            [[1, 0], [0, sqrt(1 - rate)]], dtype=DEFAULT_MATRIX_DTYPE
        )
        K1: Tensor = torch.tensor([[0, 0], [0, sqrt(rate)]], dtype=DEFAULT_MATRIX_DTYPE)
        kraus_phase_damping: list[Tensor] = [K0, K1]
        super().__init__(kraus_phase_damping, target, rate)


class GeneralizedAmplitudeDamping(Noise):
    """
    Initialize the GeneralizeAmplitudeDamping gate.

    The generalize amplitude damping channel is defined as:

    .. math::
        \\rho \\Rightarrow K_0 \\rho K_0^{\\dagger} + K_1 \\rho K_1^{\\dagger}
            + K_2 \\rho K_2^{\\dagger} + K_3 \\rho K_3^{\\dagger}

    with:

    .. code-block:: python

        K0 = sqrt(p) * [[1, 0], [0, sqrt(1 - rate)]]
        K1 = sqrt(p) * [[0, sqrt(rate)], [0, 0]]
        K2 = sqrt(1-p) * [[sqrt(1 - rate), 0], [0, 1]]
        K3 = sqrt(1-p) * [[0, 0], [sqrt(rate), 0]]

    Args:
        target (int): The index of the qubit being affected by the noise.
        error_probabilities (tuple[float,...]): The first float must be the probability
            of amplitude damping error, and the second float is the damping rate, indicating
            the probability of generalized amplitude damping.

    Raises:
        TypeError: If the probability or rate values is not float.
        ValueError: If the damping rate  a correct probability.
    """

    def __init__(
        self,
        target: int,
        error_probability: tuple[float, ...],
    ):
        probability = error_probability[0]
        rate = error_probability[1]
        if probability > 1.0 or probability < 0.0:
            raise ValueError("The probability value is not a correct probability")
        if rate > 1.0 or rate < 0.0:
            raise ValueError("The damping rate is not a correct probability")
        K0: Tensor = sqrt(probability) * torch.tensor(
            [[1, 0], [0, sqrt(1 - rate)]], dtype=DEFAULT_MATRIX_DTYPE
        )
        K1: Tensor = sqrt(probability) * torch.tensor(
            [[0, sqrt(rate)], [0, 0]], dtype=DEFAULT_MATRIX_DTYPE
        )
        K2: Tensor = sqrt(1.0 - probability) * torch.tensor(
            [[sqrt(1 - rate), 0], [0, 1]], dtype=DEFAULT_MATRIX_DTYPE
        )
        K3: Tensor = sqrt(1.0 - probability) * torch.tensor(
            [[0, 0], [sqrt(rate), 0]], dtype=DEFAULT_MATRIX_DTYPE
        )
        kraus_generalized_amplitude_damping: list[Tensor] = [K0, K1, K2, K3]
        super().__init__(kraus_generalized_amplitude_damping, target, error_probability)
