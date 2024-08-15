from __future__ import annotations

import logging
from collections import OrderedDict
from functools import cached_property
from logging import getLogger
from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import ModuleList, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import Sequence
from pyqtorch.embed import Embedding
from pyqtorch.primitives import Primitive
from pyqtorch.quantum_operation import QuantumOperation
from pyqtorch.utils import (
    ATOL,
    Operator,
    State,
    StrEnum,
    _round_operator,
    expand_operator,
    is_diag,
)

BATCH_DIM = 2
TGenerator = Union[Tensor, str, QuantumOperation, Sequence]

logger = getLogger(__name__)


def forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Forward complete")
    torch.cuda.nvtx.range_pop()


def pre_forward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executing forward")
    torch.cuda.nvtx.range_push("HamiltonianEvolution.forward")


def backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Backward complete")
    torch.cuda.nvtx.range_pop()


def pre_backward_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    logger.debug("Executed backward")
    torch.cuda.nvtx.range_push("Hamiltonian Evolution.backward")


class GeneratorType(StrEnum):
    """
    Options for types of generators allowed in HamiltonianEvolution.
    """

    PARAMETRIC_OPERATION = "parametric_operation"
    """Generators of type Primitive or Sequence which contain
       possibly trainable or non-trainable parameters."""
    OPERATION = "operation"
    """Generators of type Primitive or Sequence which do not contain parameters or contain
       constants as Parameters for example pyq.Scale(Z(0), torch.tensor([1.]))."""
    TENSOR = "tensor"
    """Generators of type torch.Tensor in which case a qubit_support needs to be passed."""
    SYMBOL = "symbol"
    """Generators which are symbolic, i.e. will be passed via the 'values' dict by the user."""


def is_diag_hamiltonian(hamiltonian: Operator, atol: Tensor = ATOL) -> bool:
    """
    Returns True if the batched tensors H are diagonal.

    Arguments:
        H: Input tensors.
        atol: Tolerance for near-zero values.

    Returns:
        True if diagonal, else False.
    """
    diag_check = torch.tensor(
        [
            is_diag(hamiltonian[..., i], atol)
            for i in range(hamiltonian.shape[BATCH_DIM])
        ],
        device=hamiltonian.device,
    )
    return bool(torch.prod(diag_check))


def evolve(hamiltonian: Operator, time_evolution: Tensor) -> Operator:
    """Get the evolved operator.

    For a hamiltonian :math:`H` and a time evolution :math:`t`, returns :math:`exp(-i H, t)`

    Arguments:
        hamiltonian: The operator :math:`H` for evolution.
        time_evolution: The evolution time :math:`t`.

    Returns:
        The evolution operator.
    """
    if is_diag_hamiltonian(hamiltonian):
        evol_operator = torch.diagonal(hamiltonian) * (-1j * time_evolution).view(
            (-1, 1)
        )
        evol_operator = torch.diag_embed(torch.exp(evol_operator))
    else:
        evol_operator = torch.transpose(hamiltonian, 0, -1) * (
            -1j * time_evolution
        ).view((-1, 1, 1))
        evol_operator = torch.linalg.matrix_exp(evol_operator)
    return torch.transpose(evol_operator, 0, -1)


class HamiltonianEvolution(Sequence):
    """
    The HamiltonianEvolution corresponds to :math:`t`, returns :math:`exp(-i H, t)` where
    a hamiltonian/generator :math:`H` and a time evolution :math:`t` are given.

    Note that the quantity :math:`H.t` is considered dimensionless.

    We can create such operation by passing different generator types:
        - A tensor representation of the generator,
        - A string when we consider the generator as a symbol.
        - Operations as a single primitive or a sequence, parameterized or not.

    Attributes:
        generator_type: This attribute informs on how the generator is inputed
        and sets the logic for applying hamiltonian evolution.
        time: The evolution time :math:`t`.
        operations: List of operations.
        cache_length: LRU cache cache_length evolution operators for given set
                    of parameter values.
    """

    def __init__(
        self,
        generator: TGenerator,
        time: Tensor | str,
        qubit_support: Tuple[int, ...] | None = None,
        generator_parametric: bool = False,
        cache_length: int = 1,
    ):
        """Initializes the HamiltonianEvolution.
        Depending on the generator argument, set the type and set the right generator getter.

        Arguments:
            generator: The generator :math:`H`.
            time: The evolution time :math:`t`.
            qubit_support: The qubits the operator acts on.
            generator_parametric: Whether the generator is parametric or not.
        """

        if isinstance(generator, Tensor):
            if qubit_support is None:
                raise ValueError(
                    "When using a Tensor generator, please pass a qubit_support."
                )
            if len(generator.shape) < 3:
                generator = generator.unsqueeze(2)
            generator = [Primitive(generator, qubit_support)]
            self.generator_type = GeneratorType.TENSOR

        elif isinstance(generator, str):
            if qubit_support is None:
                raise ValueError(
                    "When using a symbolic generator, please pass a qubit_support."
                )
            self.generator_type = GeneratorType.SYMBOL
            self.generator_symbol = generator
            generator = []
        elif isinstance(generator, (QuantumOperation, Sequence)):
            if qubit_support is not None:
                logger.warning(
                    "Taking support from generator and ignoring qubit_support input."
                )
            qubit_support = generator.qubit_support
            if generator_parametric:
                generator = [generator]
                self.generator_type = GeneratorType.PARAMETRIC_OPERATION
            else:
                generator = [
                    Primitive(
                        generator.tensor(),
                        generator.qubit_support,
                    )
                ]

                self.generator_type = GeneratorType.OPERATION
        else:
            raise TypeError(
                f"Received generator of type {type(generator)},\
                            allowed types are: [Tensor, str, Primitive, Sequence]"
            )
        super().__init__(generator)
        self._qubit_support = qubit_support  # type: ignore
        self.time = time
        logger.debug("Hamiltonian Evolution initialized")
        if logger.isEnabledFor(logging.DEBUG):
            # When Debugging let's add logging and NVTX markers
            # WARNING: incurs performance penalty
            self.register_forward_hook(forward_hook, always_call=True)
            self.register_full_backward_hook(backward_hook)
            self.register_forward_pre_hook(pre_forward_hook)
            self.register_full_backward_pre_hook(pre_backward_hook)

        self._generator_map: dict[GeneratorType, Callable] = {
            GeneratorType.SYMBOL: self._symbolic_generator,
            GeneratorType.TENSOR: self._tensor_generator,
            GeneratorType.OPERATION: self._tensor_generator,
            GeneratorType.PARAMETRIC_OPERATION: self._tensor_generator,
        }

        # to avoid recomputing hamiltonians and evolution
        self._cache_hamiltonian_evo: dict[str, Tensor] = dict()
        self.cache_length = cache_length

    @property
    def generator(self) -> ModuleList:
        """Returns the operations making the generator.

        Returns:
            The generator as a ModuleList.
        """
        return self.operations

    def flatten(self) -> ModuleList:
        return ModuleList([self])

    @property
    def param_name(self) -> Tensor | str:
        return self.time

    def _symbolic_generator(
        self,
        values: dict,
        embedding: Embedding | None = None,
    ) -> Operator:
        """Returns the generator for the SYMBOL case.

        Arguments:
            values:

        Returns:
            The generator as a tensor.
        """
        hamiltonian = values[self.generator_symbol]
        # add batch dim
        if len(hamiltonian.shape) == 2:
            return hamiltonian.unsqueeze(2)
        # cases when the batchdim is at index 0 instead of 2
        if len(hamiltonian.shape) == 3 and (
            hamiltonian.shape[0] != hamiltonian.shape[1]
        ):
            return torch.transpose(hamiltonian, 0, 2)
        if len(hamiltonian.shape) == 4 and (
            hamiltonian.shape[0] != hamiltonian.shape[1]
        ):
            return torch.permute(hamiltonian.squeeze(3), (1, 2, 0))
        return hamiltonian

    def _tensor_generator(
        self, values: dict = dict(), embedding: Embedding | None = None
    ) -> Operator:
        """Returns the generator for the TENSOR, OPERATION and PARAMETRIC_OPERATION cases.

        Arguments:
            values: Values dict with any needed parameters.

        Returns:
            The generator as a tensor.
        """
        return self.generator[0].tensor(values, embedding)

    @property
    def create_hamiltonian(self) -> Callable[[dict], Operator]:
        """A utility method for setting the right generator getter depending on the init case.

        Returns:
            The right generator getter.
        """
        return self._generator_map[self.generator_type]

    @cached_property
    def eigenvals_generator(self) -> Tensor:
        """Get eigenvalues of the underlying hamiltonian.

        Note: Only works for GeneratorType.TENSOR
        or GeneratorType.OPERATION.

        Returns:
            Eigenvalues of the operation.
        """

        return self.generator[0].eigenvalues

    @cached_property
    def spectral_gap(self) -> Tensor:
        """Difference between the moduli of the two largest eigenvalues of the generator.

        Returns:
            Tensor: Spectral gap value.
        """
        spectrum = self.eigenvals_generator
        diffs = spectrum - spectrum.T
        diffs = _round_operator(diffs)
        spectral_gap = torch.unique(torch.abs(torch.tril(diffs)))
        return spectral_gap[spectral_gap.nonzero()]

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding | None = None,
    ) -> State:
        """
        Apply the hamiltonian evolution with input parameter values.

        Arguments:
            state: Input state.
            values: Values of parameters.

        Returns:
            The transformed state.
        """

        evolved_op = self.tensor(values, embedding)
        return apply_operator(
            state=state,
            operator=evolved_op,
            qubit_support=self.qubit_support,
        )

    def tensor(
        self,
        values: dict = dict(),
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
    ) -> Operator:
        """Get the corresponding unitary over n_qubits.

        To avoid computing the evolution operator, we store it in cache wrt values.

        Arguments:
            values: Parameter value.
            Can be higher than the number of qubit support.

        Returns:
            The unitary representation.
        """
        values_cache_key = str(OrderedDict(values))
        if self.cache_length > 0 and values_cache_key in self._cache_hamiltonian_evo:
            evolved_op = self._cache_hamiltonian_evo[values_cache_key]
        else:
            hamiltonian: torch.Tensor = self.create_hamiltonian(values, embedding)  # type: ignore [call-arg]
            time_evolution: torch.Tensor = (
                values[self.time] if isinstance(self.time, str) else self.time
            )  # If `self.time` is a string / hence, a Parameter,
            # we expect the user to pass it in the `values` dict
            evolved_op = evolve(hamiltonian, time_evolution)
            nb_cached = len(self._cache_hamiltonian_evo)

            # LRU caching
            if (nb_cached > 0) and (nb_cached == self.cache_length):
                self._cache_hamiltonian_evo.pop(next(iter(self._cache_hamiltonian_evo)))
            if nb_cached < self.cache_length:
                self._cache_hamiltonian_evo[values_cache_key] = evolved_op

        if full_support is None:
            return evolved_op
        else:
            return expand_operator(evolved_op, self.qubit_support, full_support)
