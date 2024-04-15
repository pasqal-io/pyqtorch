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

        self.target = target
        self.qubit_support: Tuple[int, ...] = (self.target,)

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
                if not isinstance(proba, float):
                    raise TypeError("The probability values must be float")
                if proba > 1.0 or proba < 0.0:
                    raise ValueError("The probability values are not correct probabilities")

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

    def unitary(self) -> Tensor:
        """
        Create a batch of unitary operators from a single operator.
        Since PyQ expects tensor.Size([2**n_qubits, 2**n_qubits,batch_size]).

        Args:
            kraus_op (Tensor): The single unitary operator tensor.

        Returns:
            Tensor: A tensor containing batched unitary operators.

        Raises:
            TypeError: If the input is not a Tensor.
        """
        return [kraus_op.unsqueeze(2) for kraus_op in self.kraus]

    def dagger(self) -> Tensor:
        """
        Computes the conjugate transpose (dagger) of a Kraus operator.

        Args:
            kraus_op (Tensor): The tensor representing a quantum Kraus operator.

        Returns:
            Tensor: The conjugate transpose (dagger) of the input tensor.
        """
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
        kraus_unit = self.unitary()
        kraus_dag = self.dagger()
        for index in range (len(self.kraus)):
            rho_i: Tensor = apply_ope_ope(
                kraus_unit[index], apply_ope_ope(rho, kraus_dag[index], self.target), self.target
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
    def __init__(
        self,
        target: int,
        rate: float,  # Maybe treat != the rate and the proba
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
    def __init__(
        self,
        target: int,
        rate: float,  # Maybe treat != the rate and the proba
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
    def __init__(
        self,
        target: int,
        probability: float,
        rate: float,  # Rate remplace proba
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
