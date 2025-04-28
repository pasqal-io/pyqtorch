from __future__ import annotations

from functools import reduce
from logging import getLogger
from operator import add
from typing import Any, Generator, NoReturn, Union

import torch
from torch import Tensor, einsum, rand
from torch.nn import Module, ModuleList, ParameterDict

from pyqtorch.apply import apply_operator, apply_operator_dm
from pyqtorch.embed import ConcretizedCallable, Embedding
from pyqtorch.matrices import add_batch_dim
from pyqtorch.primitives import CNOT, RX, RY, I, Parametric, Primitive, X, Y, Z
from pyqtorch.utils import (
    DensityMatrix,
    Operator,
    State,
)

from .sequence import Sequence

BATCH_DIM = 2
PAULI_OPS = (X, Y, Z, I)

logger = getLogger(__name__)


class Scale(Sequence):
    """
    Generic container for multiplying a 'Primitive', 'Sequence' or 'Add' instance by a parameter.

    Attributes:
        operations: Operations as a Sequence, Add, or a single Primitive operation.
        param_name: Name of the parameter to multiply operations with.
    """

    def __init__(
        self,
        operations: Union[Primitive, Sequence, Add],
        param_name: str | float | int | Tensor | ConcretizedCallable,
    ):
        """
        Initializes a Scale object.

        Arguments:
            operations: Operations as a Sequence, Add, or a single Primitive operation.
            param_name: Name of the parameter to multiply operations with.
        """
        if not isinstance(operations, (Primitive, Sequence, Add)):
            raise ValueError("Scale only supports a single operation, Sequence or Add.")
        if not isinstance(param_name, (str, int, float, Tensor, ConcretizedCallable)):
            raise TypeError(
                "Only str, int, float, Tensor or ConcretizedCallable types \
                are supported for param_name"
            )
        self.is_diagonal = operations.is_diagonal
        super().__init__([operations])
        self.param_name = param_name

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | ParameterDict | None = None,
        embedding: Embedding | None = None,
    ) -> State:
        """
        Apply the operation(s) multiplying by the parameter value.

        Arguments:
            state: Input state.
            values: Parameter value.

        Returns:
            The transformed state.
        """
        values = values or dict()
        if embedding is not None:
            values = embedding(values)

        if isinstance(self.param_name, str):
            scale = values[self.param_name]
        elif isinstance(self.param_name, Tensor):
            scale = self.param_name
        elif isinstance(self.param_name, ConcretizedCallable):
            scale = self.param_name(values)

        return scale * self.operations[0].forward(state, values)

    def tensor(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
        diagonal: bool = False,
    ) -> Operator:
        """
        Get the corresponding unitary over n_qubits.

        Arguments:
            values: Parameter value.
            embedding: An optional embedding.
            full_support: Can be higher than the number of qubit support.

        Returns:
            The unitary representation.
        """
        values = values or dict()
        if embedding is not None:
            values = embedding(values)
        use_diagonal = diagonal and self.is_diagonal
        if isinstance(self.param_name, str):
            scale = values[self.param_name]
        elif isinstance(self.param_name, (Tensor, int, float)):
            scale = self.param_name
        elif isinstance(self.param_name, ConcretizedCallable):
            scale = self.param_name(values)

        return scale * self.operations[0].tensor(
            values, full_support=full_support, diagonal=use_diagonal
        )

    def flatten(self) -> list[Scale]:
        """This method should only be called in the AdjointExpectation,
        where the `Scale` is only supported for Primitive (and not Sequences)
        so we don't want to flatten this to preserve the scale parameter.

        Returns:
            The Scale within a list.
        """
        return [self]

    def to(self, *args: Any, **kwargs: Any) -> Scale:
        """Perform conversions for dtype or device.

        Returns:
            Converted Scale.
        """
        super().to(*args, **kwargs)
        if not isinstance(self.param_name, (str, float, int)):
            self.param_name = self.param_name.to(*args, **kwargs)

        return self


class Add(Sequence):
    """
    The 'add' operation applies all 'operations' to 'state' and returns the sum of states.

    Attributes:
        operations: List of operations to add up.
    """

    def __init__(self, operations: list[Module]):
        super().__init__(operations=operations)

    def forward(
        self,
        state: State,
        values: dict[str, Tensor] | ParameterDict | None = None,
        embedding: Embedding | None = None,
    ) -> State:
        """
        Apply the operations multiplying by the parameter values.

        Arguments:
            state: Input state.
            values: Parameter value.

        Returns:
            The transformed state.
        """
        values = values or dict()
        return reduce(add, (op(state, values, embedding) for op in self.operations))

    def tensor(
        self,
        values: dict | None = None,
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        """
        Get the corresponding sum of unitaries over n_qubits.

        Arguments:
            values: Parameter value.
            Can be higher than the number of qubit support.


        Returns:
            The unitary representation.
        """
        values = values or dict()
        use_diagonal = diagonal and self.is_diagonal
        if full_support is None:
            full_support = self.qubit_support
        elif not set(self.qubit_support).issubset(set(full_support)):
            raise ValueError(
                "Expanding tensor operation requires a `full_support` argument "
                "larger than or equal to the `qubit_support`."
            )
        if not use_diagonal:
            mat = torch.zeros(
                (2 ** len(full_support), 2 ** len(full_support), 1), device=self.device
            )
        else:
            mat = torch.zeros((2 ** len(full_support), 1), device=self.device)
        return reduce(
            add,
            (
                op.tensor(values, embedding, full_support, diagonal=use_diagonal)
                for op in self.operations
            ),
            mat,
        )

    @property
    def is_pauli_add(self) -> bool:
        """Check if Add contains only pauli string, that is a weighted
        sum of tensor products of pauli operators.

        Returns:
            bool: True if pauli string.
        """
        return all(isinstance(op, PAULI_OPS) for op in self._flatten())

    def _symplectic_commute(self) -> bool:
        """Check if operator is composed of pauli strings, each commuting with each other."""

        if self.is_pauli_add:
            qubit_support = self.qubit_support
            nb_support = len(qubit_support)

            # build string representation of pauli operations
            pauli_strings = [["I"] * nb_support for _ in range(len(self.operations))]
            for ind_op, op in enumerate(self.operations):
                if isinstance(op, Sequence):
                    for subop in op._flatten():
                        pauli_strings[ind_op][
                            qubit_support.index(subop.qubit_support[0])
                        ] = subop.name
                else:
                    pauli_strings[ind_op][
                        qubit_support.index(op.qubit_support[0])
                    ] = op.name

            # Convert to binary vectors
            binary_vectors = []
            for p_string in pauli_strings:
                x_vec = [1 if p in ["X", "Y"] else 0 for p in p_string]
                z_vec = [1 if p in ["Z", "Y"] else 0 for p in p_string]
                binary_vectors.append(x_vec + z_vec)  # Concatenate x and z vectors

            # Check commutation using symplectic inner product
            n_strings = len(pauli_strings)
            for i in range(n_strings):
                for j in range(i + 1, n_strings):
                    vec_i = binary_vectors[i]
                    vec_j = binary_vectors[j]

                    # Split into x and z parts
                    x_i = vec_i[:nb_support]
                    z_i = vec_i[nb_support:]
                    x_j = vec_j[:nb_support]
                    z_j = vec_j[nb_support:]

                    # Symplectic product: (x_i · z_j) XOR (z_i · x_j)
                    product = 0
                    for k in range(nb_support):
                        product ^= (x_i[k] & z_j[k]) ^ (z_i[k] & x_j[k])

                    if product == 1:
                        return False  # Found a pair that doesn't commute

            return True

        return False

    def _different_support_operations(self):
        qubit_supports_intersect = [set(op.qubit_support) for op in self.operations]

        disjoint_sets: set[int] = set()
        for s in qubit_supports_intersect:
            if s.isdisjoint(disjoint_sets):
                disjoint_sets.update(s)
            else:
                return False
        if len(disjoint_sets) == len(self.qubit_support):
            return True

        return False

    @property
    def commuting_terms(self) -> bool:
        """Return if all operations are commuting terms."""

        # when operators are defined on different qubit supports
        if len(self.operations) == 1:
            return False

        if self._symplectic_commute() or self._different_support_operations():
            return True

        return False


class Merge(Sequence):
    def __init__(
        self,
        operations: list[Module],
    ):
        """
        Merge a sequence of single qubit operations acting on the same qubit into a single
        einsum operation.

        Arguments:
            operations: A list of single qubit operations.

        """
        if (
            isinstance(operations, (list, ModuleList))
            and all([isinstance(op, (Primitive, Parametric)) for op in operations])
            and all(list([len(op.qubit_support) == 1 for op in operations]))
            and len(list(set([op.qubit_support[0] for op in operations]))) == 1
        ):
            # We want all operations to act on the same qubit

            super().__init__(operations)
            self.qubits = operations[0].qubit_support
        else:
            raise TypeError(
                f"Require all operations to act on a single qubit. Got: {operations}."
            )

        self._contains_noise = sum([op.noise is not None for op in self.operations])

    def forward(
        self,
        state: Tensor,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
    ) -> Tensor:
        batch_size = state.shape[-1]
        values = values or dict()
        if values:
            batch_size = max(
                batch_size,
                max(
                    list(
                        map(
                            lambda t: len(t) if isinstance(t, Tensor) else 1,
                            values.values(),
                        )
                    )
                ),
            )

        if self._contains_noise:
            # noisy cannot use merged in tensors, fall back to super forward
            return super().forward(state, values, embedding)

        if isinstance(state, DensityMatrix):
            return apply_operator_dm(
                state,
                add_batch_dim(self.tensor(values, embedding), batch_size),
                self.qubits,
            )

        return apply_operator(
            state,
            add_batch_dim(self.tensor(values, embedding), batch_size),
            self.qubits,
        )

    def tensor(
        self,
        values: dict[str, Tensor] | None = None,
        embedding: Embedding | None = None,
        full_support: tuple[int, ...] | None = None,
        diagonal: bool = False,
    ) -> Tensor:
        # We reverse the list of tensors here since matmul is not commutative.
        values = values or dict()
        use_diagonal = diagonal and self.is_diagonal
        if not use_diagonal:
            return reduce(
                lambda u0, u1: einsum("ijb,jkb->ikb", u0, u1),
                (
                    op.tensor(values, embedding, full_support)
                    for op in reversed(self.operations)
                ),
            )
        else:
            return reduce(
                lambda u0, u1: einsum("jb,jb->jb", u0, u1),
                (
                    op.tensor(values, embedding, full_support, diagonal=True)
                    for op in reversed(self.operations)
                ),
            )


def hea(n_qubits: int, depth: int, param_name: str) -> tuple[ModuleList, ParameterDict]:
    def _idx() -> Generator[int, Any, NoReturn]:
        i = 0
        while True:
            yield i
            i += 1

    def idxer() -> Generator[int, Any, None]:
        yield from _idx()

    idx = idxer()
    ops = []
    for _ in range(depth):
        layer = []
        for i in range(n_qubits):
            layer += [
                Merge([fn(i, f"{param_name}_{next(idx)}") for fn in [RX, RY, RX]])
            ]
        ops += layer
        ops += [
            Sequence([CNOT(i % n_qubits, (i + 1) % n_qubits) for i in range(n_qubits)])
        ]
    params = ParameterDict(
        {f"{param_name}_{n}": rand(1, requires_grad=True) for n in range(next(idx))}
    )
    return ops, params
