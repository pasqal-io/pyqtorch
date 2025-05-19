from __future__ import annotations

import logging
from collections import OrderedDict
from functools import cached_property
from logging import getLogger
from typing import Callable, Tuple, Union
from uuid import uuid4

import torch
from torch import Tensor
from torch.nn import ModuleList, ParameterDict

from pyqtorch.apply import apply_operator
from pyqtorch.circuit import Sequence
from pyqtorch.composite import Add, Scale
from pyqtorch.embed import ConcretizedCallable, Embedding
from pyqtorch.noise import AnalogNoise
from pyqtorch.primitives import Primitive
from pyqtorch.quantum_operation import QuantumOperation
from pyqtorch.time_dependent.sesolve import sesolve
from pyqtorch.utils import (
    Operator,
    SolverType,
    State,
    StrEnum,
    _round_operator,
    expand_operator,
    finitediff,
    is_diag_batched,
    is_parametric,
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
    PARAMETRIC_COMMUTING_SEQUENCE = "parametric_commuting_sequence"
    """Parametric generators of type Add where each operation commute with each other."""
    COMMUTING_SEQUENCE = "commuting_sequence"
    """Generators of type Add where each operation commute with each other."""
    TENSOR = "tensor"
    """Generators of type torch.Tensor in which case a qubit_support needs to be passed."""
    SYMBOL = "symbol"
    """Generators which are symbolic, i.e. will be passed via the 'values' dict by the user."""


COMMUTING = (
    GeneratorType.PARAMETRIC_COMMUTING_SEQUENCE,
    GeneratorType.COMMUTING_SEQUENCE,
)


def evolve(
    hamiltonian: Operator, time_evolution: Tensor, diagonal: bool = False
) -> Operator:
    """Get the evolved operator.

    For a hamiltonian :math:`H` and a time evolution :math:`t`, returns :math:`exp(-i H, t)`

    Arguments:
        hamiltonian: The operator :math:`H` for evolution.
        time_evolution: The evolution time :math:`t`.
        diagonal: Whether the hamiltonian is diagonal or not.
            Note for diagonal cases, we return a dense tensor.

    Returns:
        The evolution operator.
    """
    if diagonal:
        evol_operator = torch.transpose(hamiltonian, 0, 1) * (
            -1j * time_evolution
        ).view((-1, 1))
        evol_operator = torch.diag_embed(torch.exp(evol_operator))
    # for case 3D tensor is passed as generator
    elif is_diag_batched(hamiltonian):
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
        duration: Total duration for evolving when using a solver.
        steps: Number of steps to use when using solver.
        solver: Time-dependent Lindblad master equation solver.
        noise_operators: List of tensors or Kraus operators adding analog noise
            when solving with a Shrodinger equation solver.
    """

    def __init__(
        self,
        generator: TGenerator,
        time: Tensor | str | ConcretizedCallable,
        qubit_support: Tuple[int, ...] | None = None,
        cache_length: int = 1,
        duration: Tensor | str | float | None = None,
        steps: int = 100,
        solver: SolverType = SolverType.DP5_SE,
        use_sparse: bool = False,
        noise: list[Tensor] | AnalogNoise | None = None,
    ):
        """Initializes the HamiltonianEvolution.
        Depending on the generator argument, set the type and set the right generator getter.

        Arguments:
            generator: The generator :math:`H`.
            time: The evolution time :math:`t`.
            qubit_support: The qubits the operator acts on. If generator is a quantum
                operation or sequence of operations,
                it will be inferred from the generator.
            cache_length: LRU cache cache_length evolution operators for given set
                    of parameter values.
            duration: Total duration for evolving when using a solver.
            steps: Number of steps to use when using solver.
            solver: Time-dependent Lindblad master equation solver.
            noise: List of jump operatoes for noisy simulations or an AnalogNoise
                when solving with a Shrodinger equation solver.
        """

        self.solver_type = solver
        self.steps = steps
        self.duration = duration
        self.use_sparse = use_sparse
        self.is_diagonal = False
        self._param_uuid = str(uuid4())
        original_generator = None

        if isinstance(duration, (str, float, Tensor)) or duration is None:
            self.duration = duration
        else:
            raise ValueError(
                "Optional argument `duration` should be passed as str, float or Tensor."
            )

        if isinstance(time, (str, Tensor, ConcretizedCallable)):
            self.time = time
        else:
            raise ValueError(
                "Argument `time` should be passed as str, Tensor or ConcretizedCallable."
            )

        if isinstance(generator, Tensor):
            if qubit_support is None:
                raise ValueError(
                    "When using a Tensor generator, please pass a qubit_support."
                )
            if len(generator.shape) < 3:
                generator = generator.unsqueeze(2)
            generator = [Primitive(generator, qubit_support)]
            self.is_diagonal = generator[0].is_diagonal
            self.generator_type = GeneratorType.TENSOR

        elif isinstance(generator, str):
            if qubit_support is None:
                raise ValueError(
                    "When using a symbolic generator, please pass a qubit_support."
                )
            self.generator_type = GeneratorType.SYMBOL
            self.generator_symbol = generator
            generator = []
        elif (
            noise is None and isinstance(generator, Add) and generator.commuting_terms()
        ):
            qubit_support = generator.qubit_support
            self.generator_type = (
                GeneratorType.PARAMETRIC_COMMUTING_SEQUENCE
                if is_parametric(generator)
                else GeneratorType.COMMUTING_SEQUENCE
            )
            original_generator = generator
            generator = [
                HamiltonianEvolution(
                    op,
                    time,
                    cache_length=cache_length,
                    duration=duration,
                    steps=steps,
                    solver=solver,
                    use_sparse=use_sparse,
                    noise=noise,
                )
                for op in generator.operations
            ]
            for gen in generator:
                gen._param_uuid = self._param_uuid
            self.is_diagonal = all(gen.is_diagonal for gen in generator)

        elif isinstance(generator, (QuantumOperation, Sequence)):
            qubit_support = generator.qubit_support

            if is_parametric(generator):
                generator = [generator]
                self.generator_type = GeneratorType.PARAMETRIC_OPERATION
            else:
                # avoiding using dense tensor for diagonal generators
                tgen = generator.tensor(diagonal=generator.is_diagonal)
                generator = [
                    Primitive(
                        tgen, generator.qubit_support, diagonal=(len(tgen.size()) == 2)
                    )
                ]
                self.is_diagonal = generator[0].is_diagonal
                self.generator_type = GeneratorType.OPERATION
        else:
            raise TypeError(
                f"Received generator of type {type(generator)},\
                            allowed types are: [Tensor, str, Primitive, Sequence]"
            )
        super().__init__(generator)
        self._qubit_support = qubit_support  # type: ignore
        self._original_generator = original_generator

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
            GeneratorType.TENSOR: self._generator,
            GeneratorType.OPERATION: self._generator,
            GeneratorType.PARAMETRIC_OPERATION: self._generator,
            GeneratorType.PARAMETRIC_COMMUTING_SEQUENCE: self._commuting_generator,
            GeneratorType.COMMUTING_SEQUENCE: self._commuting_generator,
        }

        # to avoid recomputing hamiltonians and evolution
        self._cache_hamiltonian_evo: dict[str, Tensor] = dict()
        self.cache_length = cache_length

        if isinstance(noise, list):
            noise = AnalogNoise(noise, self.qubit_support)
        if noise is not None and set(noise.qubit_support) - set(self.qubit_support):
            raise ValueError(
                "The noise should be a subset or the same qubit support"
                "as HamiltonianEvolution."
            )
        self.noise = noise

    @property
    def generator(self) -> ModuleList:
        """Returns the operations making the generator.

        Returns:
            The generator as a ModuleList.
        """
        return self.operations

    @property
    def is_parametric(self) -> bool:
        """Check if operation is parametric, that only the time is parametrized.


        Returns:
            bool: True is `time` is str.
        """
        return isinstance(self.param_name, str)

    @property
    def is_parametric_generator(self) -> bool:
        return self.generator_type in (
            GeneratorType.SYMBOL,
            GeneratorType.PARAMETRIC_OPERATION,
            GeneratorType.PARAMETRIC_COMMUTING_SEQUENCE,
        )

    @cached_property
    def is_time_dependent(self) -> bool:
        return self._is_time_dependent(self.generator)

    def _flatten(self) -> ModuleList:
        return ModuleList([self])

    def flatten(self) -> ModuleList:
        return self._flatten()

    @property
    def param_name(self) -> Tensor | str:
        return self.time

    def _is_time_dependent(self, generator: TGenerator) -> bool:
        from pyqtorch.primitives import Parametric

        res = False
        if isinstance(self.time, Tensor):
            return res
        else:
            if isinstance(generator, (Sequence, QuantumOperation, ModuleList)):
                for m in generator.modules():
                    if isinstance(m, (Scale, Parametric)):
                        if self.time in getattr(m.param_name, "independent_args", []):
                            # param_name is a ConcretizedCallable object
                            res = True
                        elif m.param_name == self.time:
                            # param_name is a string
                            res = True
        return res

    def _symbolic_generator(
        self,
        values: dict,
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
    ) -> Operator:
        """Returns the generator for the SYMBOL case.

        Arguments:
            values: Values dict with any needed parameters.
            embedding: Embedding of parameters.
            full_support: The qubits the returned tensor
                will be defined over. Defaults to None for only using the qubit_support.

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

    def _generator(
        self,
        values: dict = dict(),
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
    ) -> Operator:
        """Returns the TENSOR, OPERATION and PARAMETRIC_OPERATION generator.

        Arguments:
            values: Values dict with any needed parameters.
            embedding: Embedding of parameters.
            full_support: The qubits the returned tensor
                will be defined over. Defaults to None for only using the qubit_support.

        Returns:
            The generator as a tensor.
        """
        return super().tensor(
            values, embedding, full_support, diagonal=self.is_diagonal
        )

    def _commuting_generator(
        self,
        values: dict = dict(),
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
    ) -> Operator:
        """Returns the COMMUTING_SEQUENCE generator.

        Arguments:
            values: Values dict with any needed parameters.
            embedding: Embedding of parameters.
            full_support: The qubits the returned tensor
                will be defined over. Defaults to None for only using the qubit_support.

        Returns:
            The generator as a tensor.
        """
        return self._original_generator.tensor(  # type: ignore [union-attr]
            values, embedding, full_support, diagonal=self.is_diagonal
        )

    @property
    def create_hamiltonian(self) -> Callable:
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
        if self.generator_type not in COMMUTING:
            return self.generator[0].eigenvalues
        else:
            blockmat = self.create_hamiltonian()
            if len(blockmat.shape) == 3:
                return torch.linalg.eigvals(blockmat.permute((2, 0, 1))).reshape(-1, 1)
            else:
                # for diagonal cases
                return blockmat

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

    def _time_evolution(self, values: dict[str, Tensor] | ParameterDict) -> Tensor:
        """Get the evolution from parameter values.

        Arguments:
            values: Values of parameters.

        Returns:
            The time evolution.
        """
        if isinstance(self.time, str):
            # note: GPSR trick when the same param_name is used in many operations
            if self._param_uuid in values.keys():
                time_evolution = values[self.time] + values[self._param_uuid]
            else:
                time_evolution = values[self.time]
        elif isinstance(self.time, ConcretizedCallable):
            time_evolution = self.time(values)
        else:
            time_evolution = self.time
        return time_evolution

    def _forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding | None = None,
    ) -> State:
        if self.generator_type not in COMMUTING:

            evolved_op = self.tensor(values, embedding)
            return apply_operator(
                state=state, operator=evolved_op, qubit_support=self.qubit_support
            )
        else:
            return self._forward_commuting_generators(state, values, embedding)

    def _forward_commuting_generators(
        self,
        state: Tensor,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding | None = None,
    ) -> State:
        """Apply the hamiltonian evolution with input parameter values when
        hamiltonian is composed of commuting terms.

        Arguments:
            state: Input state.
            values: Values of parameters.
            embedding: Embedding of parameters.

        Returns:
            The transformed state.
        """
        if embedding is not None:
            values = embedding(values)

        return super().forward(state, values, embedding)

    def _forward_time(
        self,
        state: Tensor,
        duration: float,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding = Embedding(),
    ) -> State:
        """Apply the hamiltonian evolution with input parameter values for time dependent cases.

        Arguments:
            state: Input state.
            values: Values of parameters.
            embedding: Embedding of parameters.

        Returns:
            The transformed state.
        """
        n_qubits = len(state.shape) - 1
        batch_size = state.shape[-1]
        t_grid = torch.linspace(0, duration, self.steps)
        if embedding is not None:
            values.update({embedding.tparam_name: torch.tensor(0.0)})  # type: ignore [dict-item]
            embedded_params = embedding(values)
        else:
            embedded_params = values

        def Ht(t: torch.Tensor) -> torch.Tensor:
            """Accepts a value 't' for time and returns
            a (2**n_qubits, 2**n_qubits) Hamiltonian evaluated at time 't'.
            """
            # We use the original embedded params and return a new dict
            # where we reembedded all parameters depending on time with value 't'
            if embedding is not None:
                reembedded_time_values = embedding.reembed_tparam(
                    embedded_params, torch.as_tensor(t)
                )
            else:
                values[self.time] = torch.as_tensor(t)
                reembedded_time_values = values
            return self.create_hamiltonian(
                reembedded_time_values,
                embedding,
                full_support=tuple(range(n_qubits)),
            ).squeeze(2)

        if self.noise is None:
            sol = sesolve(
                Ht,
                torch.flatten(state, start_dim=0, end_dim=-2),
                t_grid,
                self.solver_type,
                options={"use_sparse": self.use_sparse},
            )

            # Retrieve the last state of shape (2**n_qubits, batch_size)
            # and reshape
            state = sol.states[-1].reshape([2] * n_qubits + [batch_size])
        else:
            state = self.noise(
                state,
                Ht,
                t_grid,
                self.solver_type,
                options={"use_sparse": self.use_sparse},
                full_support=self.qubit_support,
            )
        return state

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | ParameterDict | None = None,
        embedding: Embedding | None = None,
    ) -> State:
        """Apply the hamiltonian evolution with input parameter values.

        Note when duration is None for the time_dependent generator cases, we fall back
        to considering forwarding as a time-independent case.

        Arguments:
            state: Input state.
            values: Values of parameters.
            embedding: Embedding of parameters.

        Returns:
            The transformed state.
        """
        values = values or dict()
        if self.is_time_dependent or (
            embedding is not None and getattr(embedding, "tparam_name", None)
        ):
            # to handle cases where a time-dependent generator becomes
            # time-independent with missing duration
            duration = (
                values[self.duration]
                if isinstance(self.duration, str)
                else self.duration
            )
            if duration is None:
                return self._forward(
                    state,
                    values,
                    embedding,
                )
            return self._forward_time(state, float(duration), values, embedding)  # type: ignore [arg-type]

        else:
            return self._forward(state, values, embedding)

    def tensor(
        self,
        values: dict | None = None,
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
        diagonal: bool = False,
    ) -> Operator:
        """Get the corresponding unitary over n_qubits.

        To avoid computing the evolution operator, we store it in cache wrt values.

        Arguments:
            values: Parameter values.
            embedding (Embedding | None, optional): Optional embedding. Defaults to None.
            full_support (tuple[int, ...] | None, optional): The qubits the returned tensor
                will be defined over. Defaults to None for only using the qubit_support.
            diagonal (bool, optional): Whether to return the diagonal form of the tensor or not.
                Defaults to False.

        Returns:
            The unitary representation.
        """
        values = values or dict()
        if embedding is not None:
            values = embedding(values)
        use_diagonal = diagonal and self.is_diagonal

        values_cache_key = str(OrderedDict(values))
        commuting_generator = self.generator_type in COMMUTING
        if commuting_generator:
            return super().tensor(values, embedding, full_support, use_diagonal)
        elif self.cache_length > 0 and values_cache_key in self._cache_hamiltonian_evo:
            evolved_op = self._cache_hamiltonian_evo[values_cache_key]

        else:
            hamiltonian: torch.Tensor = self.create_hamiltonian(values, embedding)  # type: ignore [call-arg]
            time_evolution = self._time_evolution(values)

            evolved_op = evolve(hamiltonian, time_evolution, diagonal=self.is_diagonal)
            if use_diagonal:
                evolved_op = torch.diagonal(evolved_op).T
            nb_cached = len(self._cache_hamiltonian_evo)

            # LRU caching
            if (nb_cached > 0) and (nb_cached == self.cache_length):
                self._cache_hamiltonian_evo.pop(next(iter(self._cache_hamiltonian_evo)))
            if nb_cached < self.cache_length:
                self._cache_hamiltonian_evo[values_cache_key] = evolved_op

        if full_support is None:
            return evolved_op
        else:
            return expand_operator(
                evolved_op, self.qubit_support, full_support, use_diagonal
            )

    def jacobian(
        self,
        values: dict[str, Tensor] = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        """Get the corresponding unitary of the jacobian.

        Arguments:
            values: Parameter value.

        Returns:
            The unitary representation of the jacobian.
        """
        if embedding is not None:
            values = embedding(values)

        hamiltonian: torch.Tensor = self.create_hamiltonian(values, embedding)  # type: ignore [call-arg]

        if self._param_uuid in values.keys():
            # note: GPSR trick when the same param_name is used in many operations
            val = values[self.param_name] + values[self._param_uuid]
        else:
            val = values[self.param_name]
        return finitediff(
            lambda t: evolve(hamiltonian, t, self.is_diagonal),
            val.reshape(-1, 1),
            (0,),
        )
