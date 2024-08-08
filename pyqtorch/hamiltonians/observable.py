from __future__ import annotations

from torch import Tensor
from torch.nn import Module, ParameterDict

from pyqtorch.circuit import Sequence
from pyqtorch.composite import Add
from pyqtorch.embed import Embedding
from pyqtorch.primitives import Primitive
from pyqtorch.utils import (
    inner_prod,
)


class Observable(Add):
    """
    The Observable :math:`O` represents an operator from which
    we can extract expectation values from quantum states.

    Given an input state :math:`\\ket\\rangle`, the expectation value with :math:`O` is defined as
    :math:`\\langle\\bra|O\\ket\\rangle`

    Attributes:
        operations: List of operations.
        n_qubits: Number of qubits it is defined on.
    """

    def __init__(
        self,
        operations: list[Module] | Primitive | Sequence,
    ):
        super().__init__(operations if isinstance(operations, list) else [operations])

    def expectation(
        self,
        state: Tensor,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding | None = None,
    ) -> Tensor:
        """Calculate the inner product :math:`\\langle\\bra|O\\ket\\rangle`

        Arguments:
            state: Input state.
            values: Values of parameters.

        Returns:
            The expectation value.
        """
        return inner_prod(state, self.forward(state, values, embedding)).real
