from __future__ import annotations

from math import sqrt
from typing import List, Tuple, Union

import torch
from torch import Tensor

from pyqtorch.apply import apply_ope_ope
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, IMAT, XMAT, YMAT, ZMAT, _dagger
from pyqtorch.utils import density_mat


class Noise(torch.nn.Module):
    def __init__(
        self, kraus: List[Tensor], target: int, probability: Union[Tuple[float, ...], float]
    ) -> None:
        super().__init__()
        self.target: int = target
        self.qubit_support: Tuple[int, ...] = (self.target,)
        # Verification Kraus operators:
        if not isinstance(kraus, list) or not all(isinstance(tensor, Tensor) for tensor in kraus):
            raise TypeError("The Kraus operators must be a list containing Tensor objects")
        if len(kraus) == 0:
            raise ValueError("The noisy gate must be described by the Kraus operator")
        # Create a buffer for the Kraus operators.
        self.kraus = []
        for index, tensor in enumerate(kraus):
            # Registering the tensor as a buffer with a unique name
            self.register_buffer(f"kraus_{index}", tensor)
            # Storing the tensor itself in the self.kraus list
            self.kraus.append(tensor)
        # Kraus operator's device
        if len(set(kraus.device for kraus in self.kraus)) != 1:
            raise ValueError("All tensors are not on the same device.")
        self._device = self.kraus[0].device
        self.probability: Union[Tuple[float, ...], float] = probability
        # Verification probability value:
        if isinstance(self.probability, float):
            if self.probability > 1.0 or self.probability < 0.0:
                raise ValueError("The probability value is not a correct probability")
        # Verification list probability values:
        elif isinstance(self.probability, list):
            sum_prob = sum(self.probability)
            if sum_prob > 1.0:
                raise ValueError("The sum of probabilities can't be greater 1.0")
            for proba in self.probability:
                if proba > 1.0 or proba < 0.0:
                    raise ValueError("The probability values are not correct probabilities")

    @property
    def proba(self) -> Union[Tuple[float, ...], float]:
        return self.probability

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.name == other.name and self.probabilty == other.probabilty
        else:
            return False

    def extra_repr(self) -> str:
        return f"'qubit_support'={self.qubit_support}, 'probability'={self.probability}"

    @property
    def kraus_operator(self) -> List[Tensor]:
        return self.kraus

    # Since PyQ expects tensor.Size = [2**n_qubits, 2**n_qubits,batch_size].
    def unitary(self) -> List[Tensor]:
        return [kraus_op.unsqueeze(2) for kraus_op in self.kraus]

    def dagger(self) -> List[Tensor]:
        return [_dagger(kraus_op) for kraus_op in self.unitary()]

    def forward(self, state: Tensor) -> Tensor:
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

        # Verification input type:
        if not isinstance(state, Tensor):
            raise TypeError("The input must be a Tensor")

        # Output operator initialization:
        n_qubits: int = len(state.size()) - 1
        batch_size: int = state.size(-1)
        rho_evol: Tensor = torch.zeros(
            2**n_qubits, 2**n_qubits, batch_size, dtype=DEFAULT_MATRIX_DTYPE
        )

        # Apply noisy channel on input state
        rho: Tensor = density_mat(state)
        kraus_unit: List[Tensor] = self.unitary()
        kraus_dag: List[Tensor] = self.dagger()
        for i in range(len(self.kraus)):
            rho_i: Tensor = apply_ope_ope(
                kraus_unit[i], apply_ope_ope(rho, kraus_dag[i], self.target), self.target
            )
            rho_evol += rho_i
        return rho_evol

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> Noise:
        super().to(device)
        self._device = device
        return self


class BitFlip(Noise):
    """
    Initialize the BitFlip gate.

    The bit flip channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho + p X \\rho X^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        probability (float): The probability of a bit flip error.

    Raises:
        TypeError: If the probability value is not a float.
    """

    def __init__(
        self,
        target: int,
        probability: float,
    ):
        # Verification probability type:
        if not isinstance(probability, float):
            raise TypeError("The probability values must be float")

        # Define Kraus operator:
        K0: Tensor = torch.sqrt(torch.tensor(1 - probability, dtype=DEFAULT_MATRIX_DTYPE)) * IMAT
        K1: Tensor = torch.sqrt(torch.tensor(probability, dtype=DEFAULT_MATRIX_DTYPE)) * XMAT
        Kraus_Bitflip: List[Tensor] = [K0, K1]
        super().__init__(Kraus_Bitflip, target, probability)


class PhaseFlip(Noise):
    """
    Initialize the PhaseFlip gate

    The phase flip channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-p) \\rho + p Z \\rho Z^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        probability (float): The probability of phase flip error.

    Raises:
        TypeError: If the probability value is not a float.
    """

    def __init__(
        self,
        target: int,
        probability: float,
    ):
        # Verification probability type:
        if not isinstance(probability, float):
            raise TypeError("The probability values must be float")

        # Define Kraus operator:
        K0: Tensor = torch.sqrt(torch.tensor(1 - probability, dtype=DEFAULT_MATRIX_DTYPE)) * IMAT
        K1: Tensor = torch.sqrt(torch.tensor(probability, dtype=DEFAULT_MATRIX_DTYPE)) * ZMAT
        Kraus_Phase: List[Tensor] = [K0, K1]
        super().__init__(Kraus_Phase, target, probability)


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
        probability (float): The probability of depolarizing error.

    Raises:
        TypeError: If the probability value is not a float.
    """

    def __init__(
        self,
        target: int,
        probability: float,
    ):
        # Verification probability type:
        if not isinstance(probability, float):
            raise TypeError("The probability values must be float")

        # Define Kraus operator:
        K0: Tensor = torch.sqrt(torch.tensor(1 - probability, dtype=DEFAULT_MATRIX_DTYPE)) * IMAT
        K1: Tensor = torch.sqrt(torch.tensor(probability / 3, dtype=DEFAULT_MATRIX_DTYPE)) * XMAT
        K2: Tensor = torch.sqrt(torch.tensor(probability / 3, dtype=DEFAULT_MATRIX_DTYPE)) * YMAT
        K3: Tensor = torch.sqrt(torch.tensor(probability / 3, dtype=DEFAULT_MATRIX_DTYPE)) * ZMAT
        Kraus_Depolarizing: List[Tensor] = [K0, K1, K2, K3]
        super().__init__(Kraus_Depolarizing, target, probability)


class PauliChannel(Noise):
    """
    Initialize the PauliChannel gate.

    The pauli channel is defined as:

    .. math::
        \\rho \\Rightarrow (1-probX-probY-probZ) \\rho
            + p_X X \\rho X^{\\dagger}
            + p_Y Y \\rho Y^{\\dagger}
            + p_Z Z \\rho Z^{\\dagger}

    Args:
        target (int): The index of the qubit being affected by the noise.
        probability (Tuple[float, ...]): Tuple containing probabilities of X, Y, and Z errors.

    Raises:
        TypeError: If the probability values are not provided in a tuple.
    """

    def __init__(
        self,
        target: int,
        probability: Tuple[float, ...],
    ):
        # Verification list probability type:
        if not isinstance(probability, list):
            raise TypeError("The probability values must be in a list")

        # Define Kraus operator:
        p_x = probability[0]
        p_y = probability[1]
        p_z = probability[2]
        K0: Tensor = (
            torch.sqrt(torch.tensor(1 - (p_x + p_y + p_z), dtype=DEFAULT_MATRIX_DTYPE)) * IMAT
        )
        K1: Tensor = torch.sqrt(torch.tensor(p_x, dtype=DEFAULT_MATRIX_DTYPE)) * XMAT
        K2: Tensor = torch.sqrt(torch.tensor(p_y, dtype=DEFAULT_MATRIX_DTYPE)) * YMAT
        K3: Tensor = torch.sqrt(torch.tensor(p_z, dtype=DEFAULT_MATRIX_DTYPE)) * ZMAT
        Kraus_PauliChannel: List[Tensor] = [K0, K1, K2, K3]
        super().__init__(Kraus_PauliChannel, target, probability)


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
        rate (float): The damping rate, indicating the probability of amplitude loss.

    Raises:
        TypeError: If the rate value is not a float.
        ValueError: If the damping rate is not a correct probability.
    """

    def __init__(
        self,
        target: int,
        rate: float,
    ):
        # Verification rate type and value:
        if not isinstance(rate, float):
            raise TypeError("The probability values must be float")
        if rate > 1.0 or rate < 0.0:
            raise ValueError("The damping rate is not a correct probability")

        # Define Kraus operator:
        K0: Tensor = torch.tensor([[1, 0], [0, sqrt(1 - rate)]], dtype=DEFAULT_MATRIX_DTYPE)
        K1: Tensor = torch.tensor([[0, sqrt(rate)], [0, 0]], dtype=DEFAULT_MATRIX_DTYPE)
        Kraus_AmplitudeDamping: List[Tensor] = [K0, K1]
        super().__init__(Kraus_AmplitudeDamping, target, rate)


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
        rate (float): The damping rate, indicating the probability of phase damping.

    Raises:
        TypeError: If the rate value is not a float.
        ValueError: If the damping rate is not a correct probability.
    """

    def __init__(
        self,
        target: int,
        rate: float,
    ):
        # Verification rate type and value:
        if not isinstance(rate, float):
            raise TypeError("The probability values must be float")
        if rate > 1.0 or rate < 0.0:
            raise ValueError("The damping rate is not a correct probability")

        K0: Tensor = torch.tensor([[1, 0], [0, sqrt(1 - rate)]], dtype=DEFAULT_MATRIX_DTYPE)
        K1: Tensor = torch.tensor([[0, 0], [0, sqrt(rate)]], dtype=DEFAULT_MATRIX_DTYPE)
        Kraus_GeneralizeAmplitudeDamping: List[Tensor] = [K0, K1]
        super().__init__(Kraus_GeneralizeAmplitudeDamping, target, rate)


class GeneralizeAmplitudeDamping(Noise):
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
        K1 = sqrt(1-p) * [[0, 0], [sqrt(rate), 0]]

    Args:
        target (int): The index of the qubit being affected by the noise.
        probability (float): The probability of amplitude damping error.
        rate (float): The damping rate, indicating the probability of generalized amplitude damping.

    Raises:
        TypeError: If the probability or rate values is not float.
        ValueError: If the damping rate  a correct probability.
    """

    def __init__(
        self,
        target: int,
        probability: float,
        rate: float,
    ):
        # Verification probability/rate type:
        if not isinstance(probability, float):
            raise TypeError("The probability values must be float")
        if not isinstance(rate, float):
            raise TypeError("The damping rate must be float")
        if rate > 1.0 or rate < 0.0:
            raise ValueError("The damping rate is not a correct probability")

        # Define Kraus operator:
        K0: Tensor = torch.sqrt(
            torch.tensor(probability, dtype=DEFAULT_MATRIX_DTYPE)
        ) * torch.tensor([[1, 0], [0, sqrt(1 - rate)]], dtype=DEFAULT_MATRIX_DTYPE)
        K1: Tensor = torch.sqrt(
            torch.tensor(probability, dtype=DEFAULT_MATRIX_DTYPE)
        ) * torch.tensor([[0, sqrt(rate)], [0, 0]], dtype=DEFAULT_MATRIX_DTYPE)
        K2: Tensor = torch.sqrt(
            torch.tensor(1 - probability, dtype=DEFAULT_MATRIX_DTYPE)
        ) * torch.tensor([[sqrt(1 - rate), 0], [0, 1]], dtype=DEFAULT_MATRIX_DTYPE)
        K3: Tensor = torch.sqrt(
            torch.tensor(1 - probability, dtype=DEFAULT_MATRIX_DTYPE)
        ) * torch.tensor([[0, 0], [sqrt(rate), 0]], dtype=DEFAULT_MATRIX_DTYPE)
        Kraus_GeneralizeAmplitudeDamping: List[Tensor] = [K0, K1, K2, K3]
        super().__init__(Kraus_GeneralizeAmplitudeDamping, target, probability)
