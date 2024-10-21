from __future__ import annotations

from collections import Counter
from enum import Enum

import torch
from torch import Tensor
from torch.distributions import normal, poisson, uniform

from pyqtorch.utils import OrderedCounter


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


def bit_flip(bit: Tensor, cond: Tensor) -> Tensor:
    """
    A helper function that reverses the states 0 and 1 in the bit string.

    Args:
        bit: A integer-value bit in a bitstring to be inverted.
        cond: A Bool value of whether or not a bit should be flipped.

    Returns:
        The inverse value of the input bit
    """
    return torch.where(
        cond,
        torch.where(
            bit == torch.zeros(1, dtype=torch.int64),
            torch.ones(1, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        ),
        bit,
    )


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


def bs_corruption(
    err_idx: Tensor,
    sample: Tensor,
) -> Counter:
    """
    A function that incorporates the expected readout error in a sample of bit strings.

    given a noise matrix.

    Args:
        err_idx: A Boolean array of bit string indices that need to be corrupted.
        sample: A torch.Tensor of bit strings n_shots x n_qubits.

    Returns:
        A counter of bit strings after readout corruption.
    """

    func = torch.func.vmap(bit_flip)

    return Counter([tensor_to_bitstring(k) for k in func(sample, err_idx)])


def create_confusion_matrices(noise_matrix: Tensor, error_probability: float) -> Tensor:
    confusion_matrices = []
    for i in range(noise_matrix.size()[1]):
        column_tensor = noise_matrix[:, i]
        flip_proba = column_tensor[column_tensor < error_probability].mean().item()
        confusion_matrix = torch.tensor(
            [[1.0 - flip_proba, flip_proba], [flip_proba, 1.0 - flip_proba]],
            dtype=torch.float64,
        )
        confusion_matrices.append(confusion_matrix)
    return torch.stack(confusion_matrices)


class ReadoutNoise:
    """Simulate errors when sampling from a circuit.

    The model is simple as all bits are considered independent
    and are corrupted with an equal `error_probability`.

    The simulation is done by drawing samples from a `noise_distribution`.
    These samples are then compared to `error_probability` to specify
    which bits are corrupted.


    Attributes:
        n_qubits (int): Number of qubits.
        seed (int | None, optional): Random seed value. Defaults to None.
        error_probability (float | None, optional): Uniform error probability of wrong
            readout at any position in the bit strings. Defaults to None.
        noise_distribution (str, optional): Noise distribution type. Defaults to WhiteNoise.UNIFORM.
    """

    def __init__(
        self,
        n_qubits: int,
        error_probability: float | None = None,
        seed: int | None = None,
        noise_distribution: torch.distributions = WhiteNoise.UNIFORM,
    ) -> None:
        """Initializes ReadoutNoise.

        Args:
            n_qubits (int): Number of qubits.
            seed (int | None, optional): Random seed value. Defaults to None.
            error_probability (float | None, optional): Uniform error probability of wrong
              readout at any position in the bit strings. Defaults to 0.1 if None.
            noise_distribution (str, optional): Noise distribution type.
              Defaults to WhiteNoise.UNIFORM.
        """
        self.n_qubits = n_qubits
        self.seed = seed
        self.error_probability = error_probability if error_probability else 0.1
        self.noise_distribution = noise_distribution

    def create_noise_matrix(
        self, n_shots: int, return_confusion: bool = False
    ) -> Tensor | tuple[Tensor]:
        """Create a noise matrix from a noise distribution.

        Also possibly return the confusion matrices if needed.

        Args:
            n_shots (int): Number of shots.
            return_confusion (bool, optional): If True,
            return the confusion matrices. Defaults to False.

        Returns:
            Tensor | tuple[Tensor]: The noise matrix and possibly the confusion ones too.
        """

        if self.seed is not None:
            torch.manual_seed(self.seed)
        noise_matrix = create_noise_matrix(
            self.noise_distribution, n_shots, self.n_qubits
        )
        if return_confusion:
            confusion_matrices = create_confusion_matrices(
                noise_matrix=noise_matrix, error_probability=self.error_probability
            )
            return noise_matrix, confusion_matrices
        return noise_matrix

    def apply_on_probas(self, batch_probs: Tensor, n_shots: int = 1000) -> Tensor:
        """Apply confusion matrix on probabilities.

        Args:
            batch_probs (Tensor): Batch of probability vectors.
            n_shots (int, optional): Number of shots. Defaults to 1000.

        Returns:
            Tensor: Corrupted probabilities.
        """

        # Create binary representations
        n_states = batch_probs.shape[1]

        # Create binary representation of all states
        state_indices = torch.arange(n_states, device=batch_probs.device)
        binary_repr = (
            state_indices.unsqueeze(1)
            >> torch.arange(self.n_qubits - 1, -1, -1, device=batch_probs.device)
        ) & 1

        # Get input and output bits for all qubits at once
        input_bits = binary_repr.unsqueeze(0).expand(n_states, -1, -1)
        output_bits = binary_repr.unsqueeze(1).expand(-1, n_states, -1)

        # Get transition probabilities for each bit position
        _, confusion_matrices = self.create_noise_matrix(n_shots, True)  # type: ignore[misc]

        # Index into confusion matrix for all qubits at once
        # Shape: (n_states_out, n_states_in, n_qubits)
        qubit_transitions = confusion_matrices[
            torch.arange(self.n_qubits, device=batch_probs.device),
            output_bits,
            input_bits,
        ]
        transition_matrix = torch.prod(qubit_transitions, dim=-1)

        output_probs = torch.matmul(batch_probs, transition_matrix.T)
        return output_probs

    def apply_on_counts(
        self, counters: list[Counter | OrderedCounter], n_shots: int = 1000
    ) -> list[Counter]:
        """Apply readout on counters represented as Counters.

        Args:
            counters (list[Counter  |  OrderedCounter]): Samples of bit string as Counters.
            n_shots (int, optional): Number of shots to sample. Defaults to 1000.

        Returns:
            list[Counter]: Samples of corrupted bit strings
        """
        noise_matrix = self.create_noise_matrix(n_shots, False)
        err_idx = torch.as_tensor(noise_matrix < self.error_probability)  # type: ignore[operator]

        corrupted_bitstrings = []
        for counter in counters:  # type: ignore[assignment]
            sample = sample_to_matrix(counter)
            corrupted_bitstrings.append(bs_corruption(err_idx=err_idx, sample=sample))
        return corrupted_bitstrings
