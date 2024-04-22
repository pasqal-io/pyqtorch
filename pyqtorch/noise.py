from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor

from pyqtorch.apply import operator_product
from pyqtorch.matrices import DEFAULT_MATRIX_DTYPE, IMAT, XMAT, YMAT, ZMAT, _dagger
from pyqtorch.utils import density_mat


class Noise(torch.nn.Module):
    def __init__(
        self, kraus: list[Tensor], target: int, probabilities: tuple[float, ...] | float
    ) -> None:
        super().__init__()
        self.target: int = target
        self.qubit_support: tuple[int, ...] = (self.target,)
        self.kraus = []
        for index, tensor in enumerate(kraus):
            self.register_buffer(f"kraus_{index}", tensor)
            self.kraus.append(tensor)
        if len(set(kraus.device for kraus in self.kraus)) != 1:
            raise ValueError("All tensors are not on the same device.")
        self._device = self.kraus[0].device
        self.probabilities: tuple[float, ...] | float = probabilities

    @property
    def proba(self) -> tuple[float, ...] | float:
        return self.probabilities

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.name == other.name and self.probabilty == other.probabilty
        return False

    def extra_repr(self) -> str:
        return f"'qubit_support'={self.qubit_support}, 'probabilities'={self.probabilities}"

    @property
    def kraus_operator(self) -> list[Tensor]:
        return self.kraus

    # Since PyQ expects tensor.Size = [2**n_qubits, 2**n_qubits,batch_size].
    def unitary(self) -> list[Tensor]:
        return [kraus_op.unsqueeze(2) for kraus_op in self.kraus]

    def dagger(self) -> list[Tensor]:
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
        if not isinstance(state, Tensor):
            raise TypeError("The input must be a Tensor")
        n_qubits: int = len(state.size()) - 1
        batch_size: int = state.size(-1)
        rho_evol: Tensor = torch.zeros(
            2**n_qubits, 2**n_qubits, batch_size, dtype=DEFAULT_MATRIX_DTYPE
        )
        rho: Tensor = density_mat(state)
        kraus_unit: list[Tensor] = self.unitary()
        kraus_dag: list[Tensor] = self.dagger()
        for i in range(len(self.kraus)):
            rho_i: Tensor = operator_product(
                kraus_unit[i], operator_product(rho, kraus_dag[i], self.target), self.target
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
        if probability > 1.0 or probability < 0.0:
            raise ValueError("The probability value is not a correct probability")
        proba_tensor = torch.tensor(probability, dtype=DEFAULT_MATRIX_DTYPE)
        K0: Tensor = torch.sqrt(1.0 - proba_tensor) * IMAT
        K1: Tensor = torch.sqrt(proba_tensor) * XMAT
        Kraus_Bitflip: list[Tensor] = [K0, K1]
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
        if probability > 1.0 or probability < 0.0:
            raise ValueError("The probability value is not a correct probability")
        proba_tensor = torch.tensor(probability, dtype=DEFAULT_MATRIX_DTYPE)
        K0: Tensor = torch.sqrt(1.0 - proba_tensor) * IMAT
        K1: Tensor = torch.sqrt(proba_tensor) * ZMAT
        Kraus_Phase: list[Tensor] = [K0, K1]
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
        if probability > 1.0 or probability < 0.0:
            raise ValueError("The probability value is not a correct probability")
        proba_tensor = torch.tensor(probability, dtype=DEFAULT_MATRIX_DTYPE)
        K0: Tensor = torch.sqrt(1.0 - proba_tensor) * IMAT
        K1: Tensor = torch.sqrt(proba_tensor / 3) * XMAT
        K2: Tensor = torch.sqrt(proba_tensor / 3) * YMAT
        K3: Tensor = torch.sqrt(proba_tensor / 3) * ZMAT
        Kraus_Depolarizing: list[Tensor] = [K0, K1, K2, K3]
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
        probabilities (Tuple[float, ...]): Tuple containing probabilities of X, Y, and Z errors.

    Raises:
        TypeError: If the probabilities values are not provided in a tuple.
    """

    def __init__(
        self,
        target: int,
        probabilities: tuple[float, ...],
    ):
        sum_prob = sum(probabilities)
        if sum_prob > 1.0:
            raise ValueError("The sum of probabilities can't be greater 1.0")
        for probability in probabilities:
            if probability > 1.0 or probability < 0.0:
                raise ValueError("The probability values are not correct probabilities")
        px = torch.tensor(probabilities[0], dtype=DEFAULT_MATRIX_DTYPE)
        py = torch.tensor(probabilities[1], dtype=DEFAULT_MATRIX_DTYPE)
        pz = torch.tensor(probabilities[2], dtype=DEFAULT_MATRIX_DTYPE)
        K0: Tensor = torch.sqrt(1.0 - (px + py + pz)) * IMAT
        K1: Tensor = torch.sqrt(px) * XMAT
        K2: Tensor = torch.sqrt(py) * YMAT
        K3: Tensor = torch.sqrt(pz) * ZMAT
        Kraus_PauliChannel: list[Tensor] = [K0, K1, K2, K3]
        super().__init__(Kraus_PauliChannel, target, probabilities)


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
        if rate > 1.0 or rate < 0.0:
            raise ValueError("The damping rate is not a correct probability")
        K0: Tensor = torch.tensor([[1, 0], [0, sqrt(1 - rate)]], dtype=DEFAULT_MATRIX_DTYPE)
        K1: Tensor = torch.tensor([[0, sqrt(rate)], [0, 0]], dtype=DEFAULT_MATRIX_DTYPE)
        Kraus_AmplitudeDamping: list[Tensor] = [K0, K1]
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
        if rate > 1.0 or rate < 0.0:
            raise ValueError("The damping rate is not a correct probability")
        K0: Tensor = torch.tensor([[1, 0], [0, sqrt(1 - rate)]], dtype=DEFAULT_MATRIX_DTYPE)
        K1: Tensor = torch.tensor([[0, 0], [0, sqrt(rate)]], dtype=DEFAULT_MATRIX_DTYPE)
        Kraus_GeneralizeAmplitudeDamping: list[Tensor] = [K0, K1]
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
        if probability > 1.0 or probability < 0.0:
            raise ValueError("The probability value is not a correct probability")
        if rate > 1.0 or rate < 0.0:
            raise ValueError("The damping rate is not a correct probability")
        proba_tensor = torch.tensor(probability, dtype=DEFAULT_MATRIX_DTYPE)
        K0: Tensor = torch.sqrt(proba_tensor) * torch.tensor(
            [[1, 0], [0, sqrt(1 - rate)]], dtype=DEFAULT_MATRIX_DTYPE
        )
        K1: Tensor = torch.sqrt(proba_tensor) * torch.tensor(
            [[0, sqrt(rate)], [0, 0]], dtype=DEFAULT_MATRIX_DTYPE
        )
        K2: Tensor = torch.sqrt(1.0 - proba_tensor) * torch.tensor(
            [[sqrt(1 - rate), 0], [0, 1]], dtype=DEFAULT_MATRIX_DTYPE
        )
        K3: Tensor = torch.sqrt(1.0 - proba_tensor) * torch.tensor(
            [[0, 0], [sqrt(rate), 0]], dtype=DEFAULT_MATRIX_DTYPE
        )
        Kraus_GeneralizeAmplitudeDamping: list[Tensor] = [K0, K1, K2, K3]
        super().__init__(Kraus_GeneralizeAmplitudeDamping, target, probability)
