from __future__ import annotations

from abc import ABC
from collections import Counter
from enum import Enum
from functools import singledispatchmethod
from math import log

import torch
from torch import Tensor
from torch.distributions import normal, poisson, uniform


class WhiteNoise(Enum):
    """White noise distributions."""

    UNIFORM = staticmethod(uniform.Uniform(low=0.0, high=1.0))
    """Uniform white noise."""

    GAUSSIAN = staticmethod(normal.Normal(loc=0.0, scale=1.0))
    """Gaussian white noise."""

    POISSON = staticmethod(poisson.Poisson(rate=0.1))
    """Poisson white noise."""


def bitstring_to_tensor(bitstring: str) -> Tensor:
    """
    A helper function to convert bit strings to torch.Tensor.

    Args:
        bitstring:  A str format of a bit string.

    Returns:
        A torch.Tensor out of the input bit string.
    """
    return torch.as_tensor(list(map(int, bitstring)))


def tensor_to_bitstring(bitstring: Tensor) -> str:
    """
    A helper function to convert torch.Tensor to bit strings.

    Args:
        bitstring: A torch.Tensor format of a bit string.

    Returns:
        A str out of the input bit string.
    """
    return "".join(list(map(str, bitstring.detach().tolist())))


def sample_to_matrix(sample: dict) -> Tensor:
    """
    A helper function that maps a sample dict to a bit string array.

    Args:
        sample: A dictionary with bitstrings as keys and values
        as their counts.

    Returns: A torch.Tensor of bit strings n_shots x n_qubits.
    """

    return torch.concatenate(
        list(
            map(
                lambda bitstring: torch.broadcast_to(
                    bitstring_to_tensor(bitstring), [sample[bitstring], len(bitstring)]
                ),
                sample.keys(),
            )
        )
    )


def create_noise_matrix(
    noise_distribution: torch.distributions, n_shots: int, n_qubits: int
) -> Tensor:
    """
    A helper function that creates a noise matrix for bit string corruption.

    NB: The noise matrix is not square, as all bits are considered independent.

    Args:
        noise_distribution: Torch statistical distribution one of Gaussian,
        Uniform, or Poisson.
        n_shots: Number of shots/samples.
        n_qubits: Number of qubits

    Returns:
        A sample out of the requested distribution given the number of shots/samples.
    """
    # the noise_matrix should be available to the user if they want to do error correction
    return noise_distribution.sample([n_shots, n_qubits])


def bs_bitflip_corruption(
    err_idx: Tensor,
    sample: Tensor,
) -> Counter:
    """
    A function that incorporates the expected readout error in a sample of bit strings.

    given a noise matrix.

    Args:
        err_idx (Tensor): A Boolean array of bit string indices to be corrupted.
        sample (Tensor): A torch.Tensor of bit strings n_shots x n_qubits.

    Returns:
        Counter: A counter of bit strings after readout corruption.
    """

    corrupted = sample ^ err_idx

    return Counter([tensor_to_bitstring(k) for k in corrupted])


def bs_confusion_corruption(
    confusion_matrix: Tensor,
    sample: Tensor,
) -> Counter:
    """Given a confusion matrix and samples, corrupt by multinomial samples.

    Args:
        confusion_matrix (Tensor): Confusion matrix shape 2**n_bits x 2**n_bits.
        sample (Tensor): A torch.Tensor of bit strings n_shots x n_qubits.

    Returns:
        Counter: A counter of bit strings after readout corruption.
    """
    n_bits = sample.shape[1]
    sample_indices = (
        sample * (2 ** torch.arange(n_bits - 1, -1, -1, device=sample.device).long())
    ).sum(dim=1)
    corrupted_indices = torch.multinomial(
        confusion_matrix[sample_indices], num_samples=1, replacement=True
    ).squeeze()

    corrupted_bitstrings = torch.zeros_like(sample)
    for i in range(n_bits):
        corrupted_bitstrings[:, i] = (corrupted_indices // (2 ** (n_bits - i - 1))) % 2

    return Counter([tensor_to_bitstring(k) for k in corrupted_bitstrings])


def create_confusion_matrices(noise_matrix: Tensor, error_probability: float) -> Tensor:
    confusion_matrices = []
    for i in range(noise_matrix.size()[1]):
        column_tensor = noise_matrix[:, i]
        flip_proba = (
            flip_proba.mean().item()
            if len(flip_proba := column_tensor[column_tensor < error_probability]) > 0
            else 0.0
        )
        confusion_matrix = torch.tensor(
            [[1.0 - flip_proba, flip_proba], [flip_proba, 1.0 - flip_proba]],
            dtype=torch.float64,
        )
        confusion_matrices.append(confusion_matrix)
    return torch.stack(confusion_matrices)


class ReadoutInterface(ABC):
    """Interface for readout protocols."""

    @singledispatchmethod
    def apply(self, inputs, n_shots):
        """Apply protocol on an input to corrupt.
            Can be a tensor of probabilities or a list of counters.

        Args:
            inputs: Inputs (probabilities or list of counters) to corrupt
            n_shots: Number of shots
        """
        raise NotImplementedError


class ReadoutNoise(ReadoutInterface):
    """Simulate errors when sampling from a circuit.

    The model is simple as all bits are considered independent
    and are corrupted with an equal `error_probability`.

    The simulation is done by drawing samples from a `noise_distribution`.
    These samples are then compared to `error_probability` to specify
    which bits are corrupted.


    Attributes:
        n_qubits (int): Number of qubits.
        seed (int | None, optional): Random seed value. Defaults to None.
        error_probability (float, Tensor, optional): Error probabilities of wrong
              readout. Defaults to 0.1 at any position in the bit strings for every bit if None.
              If float, the same probability is applied to every bit. If a 1D tensor of size
              n_qubits, a different probability is set for each qubit.
              If a tensor of shape (n_qubits, 2, 2) is passed, that is a confusion matrix,
              we extract the error_probability
              and do not compute the confusion as in the other cases.
        noise_distribution (str, optional): Noise distribution type. Defaults to WhiteNoise.UNIFORM.
    """

    def __init__(
        self,
        n_qubits: int,
        error_probability: float | Tensor = 0.1,
        seed: int | None = None,
        noise_distribution: torch.distributions | None = WhiteNoise.UNIFORM,
    ) -> None:
        """Initializes ReadoutNoise.

        Args:
            n_qubits (int): Number of qubits.
            seed (int | None, optional): Random seed value. Defaults to None.
            error_probability (float, Tensor, optional): Error probabilities of wrong
              readout. Defaults to 0.1 at any position in the bit strings for every bit if None.
              If float, the same probability is applied to every bit. If a 1D tensor of size
              n_qubits, a different probability is set for each qubit.
              If a tensor of shape (n_qubits, 2, 2) is passed, that is a confusion matrix,
              we extract the error_probability
              and do not compute the confusion as in the other cases.
            noise_distribution (str, optional): Noise distribution type.
              Defaults to WhiteNoise.UNIFORM.
            confusion_matrix (Tensor, optional): The confusion matrix if available.

        """
        self.n_qubits = n_qubits
        self.seed = seed
        size_error_probability = (1,)
        if isinstance(error_probability, float):
            error_probability = torch.tensor(error_probability)

        elif isinstance(error_probability, Tensor):
            size_error_probability = tuple(error_probability.size())
            if (
                len(size_error_probability) == 1
                and len(error_probability) != self.n_qubits
            ):
                raise ValueError(
                    f"`error_probability` should have {n_qubits} elements."
                )
            if (len(size_error_probability) == 3) and (
                size_error_probability != (self.n_qubits, 2, 2)
            ):
                raise ValueError(
                    f"`error_probability` should have {(n_qubits, 2, 2)} elements."
                )
        else:
            raise ValueError(
                f"`error_probability` should be float, 1D or {(n_qubits, 2, 2)} tensor."
            )

        self.error_probability = error_probability
        self.noise_distribution = noise_distribution
        self._compute_confusion: bool = (
            False if size_error_probability == (self.n_qubits, 2, 2) else True
        )

        if not self._compute_confusion:
            self.confusion_matrix = self.error_probability
            self.error_probability = torch.empty((self.n_qubits, 2))
            self.error_probability[:, 0] = self.confusion_matrix[:, 0, 1]
            self.error_probability[:, 1] = self.confusion_matrix[:, 1, 0]
        else:
            self.confusion_matrix = torch.empty((self.n_qubits, 2, 2))

    def create_noise_matrix(self, n_shots: int) -> Tensor:
        """Create a noise matrix from a noise distribution.

        Also save the confusion matrix.

        Args:
            n_shots (int): Number of shots.

        Returns:
            Tensor | tuple[Tensor]: The noise matrix and possibly the confusion ones too.
        """

        if self.seed is not None:
            torch.manual_seed(self.seed)

        noise_matrix = create_noise_matrix(
            self.noise_distribution, n_shots, self.n_qubits
        )
        if self._compute_confusion:
            confusion_matrices = create_confusion_matrices(
                noise_matrix=noise_matrix, error_probability=self.error_probability
            )
            self.confusion_matrix = confusion_matrices
        return noise_matrix

    @singledispatchmethod
    def apply(self, inputs, n_shots):
        raise NotImplementedError

    @apply.register
    def _(self, inputs: Tensor, n_shots: int) -> Tensor:
        """Apply confusion matrix on probabilities.

        Args:
            inputs (Tensor): Batch of probability vectors.
            n_shots (int, optional): Number of shots.

        Returns:
            Tensor: Corrupted probabilities.
        """

        # Create binary representations
        n_states = inputs.shape[1]

        # Create binary representation of all states
        state_indices = torch.arange(n_states, device=inputs.device)
        binary_repr = (
            state_indices.unsqueeze(1)
            >> torch.arange(self.n_qubits - 1, -1, -1, device=inputs.device)
        ) & 1

        # Get input and output bits for all qubits at once
        input_bits = binary_repr.unsqueeze(0).expand(n_states, -1, -1)
        output_bits = binary_repr.unsqueeze(1).expand(-1, n_states, -1)

        # Get transition probabilities for each bit position
        self.create_noise_matrix(n_shots)
        confusion_matrices = self.confusion_matrix

        # Index into confusion matrix for all qubits at once
        # Shape: (n_states_out, n_states_in, n_qubits)
        qubit_transitions = confusion_matrices[
            torch.arange(self.n_qubits, device=inputs.device),
            output_bits,
            input_bits,
        ]
        transition_matrix = torch.prod(qubit_transitions, dim=-1)
        output_probs = torch.matmul(inputs, transition_matrix.T)
        return output_probs

    @apply.register
    def _(self, inputs: list, n_shots: int) -> list[Counter]:
        """Apply readout on counters represented as Counters.

        Args:
            inputs (list[Counter | OrderedCounter]): Samples of bit string as Counters.
            n_shots (int, optional): Number of shots to sample. Defaults to 1000.

        Returns:
            list[Counter]: Samples of corrupted bit strings
        """
        noise_matrix = self.create_noise_matrix(n_shots)
        err_idx = torch.as_tensor(noise_matrix < self.error_probability)

        corrupted_bitstrings = []
        for counter in inputs:
            sample = sample_to_matrix(counter)
            corrupted_bitstrings.append(
                bs_bitflip_corruption(err_idx=err_idx, sample=sample)
            )
        return corrupted_bitstrings


class CorrelatedReadoutNoise(ReadoutInterface):
    def __init__(
        self,
        confusion_matrix: Tensor,
        seed: int | None = None,
    ) -> None:
        """Initializes CorrelatedReadoutNoise.

        Args:
            confusion_matrices (Tensor): Confusion matrices of size (2**n_qubits, 2**n_qubits).
        """
        if (len(confusion_matrix.size()) != 2) or (
            confusion_matrix.size(0) != confusion_matrix.size(1)
        ):
            raise ValueError("The confusion matrix should be square")
        self.confusion_matrix = confusion_matrix
        self.n_qubits = int(log(confusion_matrix.size(0), 2))
        self.seed = seed

    @singledispatchmethod
    def apply(self, inputs, n_shots):
        raise NotImplementedError

    @apply.register
    def _(self, inputs: Tensor, n_shots: int) -> Tensor:
        """Apply confusion matrix on probabilities.

        Args:
            inputs (Tensor): Batch of probability vectors.
            n_shots (int, optional): Number of shots.

        Returns:
            Tensor: Corrupted probabilities.
        """
        output_probs = inputs @ self.confusion_matrix.T
        return output_probs

    @apply.register
    def _(self, inputs: list, n_shots: int) -> list[Counter]:
        """Apply readout on counters represented as Counters.

        Args:
            inputs (list[Counter |  OrderedCounter]): Samples of bit string as Counters.
            n_shots (int, optional): Number of shots to sample. Defaults to 1000.

        Returns:
            list[Counter]: Samples of corrupted bit strings
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
        corrupted_bitstrings = []
        for counter in inputs:
            sample = sample_to_matrix(counter)
            corrupted_bitstrings.append(
                bs_confusion_corruption(self.confusion_matrix, sample)
            )
        return corrupted_bitstrings
