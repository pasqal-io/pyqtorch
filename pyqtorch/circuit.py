from __future__ import annotations

from collections import Counter
from functools import reduce
from logging import getLogger
from operator import add

import torch
from torch import Tensor, bernoulli, tensor
from torch.nn import Module, ParameterDict

from pyqtorch.composite import Sequence
from pyqtorch.embed import Embedding
from pyqtorch.utils import (
    DensityMatrix,
    DropoutMode,
    State,
    product_state,
    sample_multinomial,
    zero_state,
)

logger = getLogger(__name__)


class QuantumCircuit(Sequence):
    """A QuantumCircuit defining a register / number of qubits of the full system."""

    def __init__(self, n_qubits: int, operations: list[Module]):
        super().__init__(operations)
        self.n_qubits = n_qubits

    def run(
        self,
        state: Tensor = None,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding | None = None,
    ) -> State:
        if state is None:
            state = self.init_state()
        elif isinstance(state, str):
            state = self.state_from_bitstring(state)
        return self.forward(state, values, embedding)

    def __hash__(self) -> int:
        return hash(reduce(add, (hash(op) for op in self.operations))) + hash(
            self.n_qubits
        )

    def init_state(self, batch_size: int = 1) -> Tensor:
        return zero_state(
            self.n_qubits, batch_size, device=self.device, dtype=self.dtype
        )

    def state_from_bitstring(self, bitstring: str, batch_size: int = 1) -> Tensor:
        return product_state(bitstring, batch_size, self.device, self.dtype)

    def sample(
        self,
        state: Tensor = None,
        values: dict[str, Tensor] = dict(),
        n_shots: int = 1000,
        embedding: Embedding | None = None,
    ) -> list[Counter]:
        if n_shots < 1:
            raise ValueError(
                f"You can only call sample with a non-negative value for `n_shots`. Got {n_shots}."
            )

        with torch.no_grad():
            state = self.run(state=state, values=values, embedding=embedding)
            if isinstance(state, DensityMatrix):
                probs = torch.diagonal(state, dim1=0, dim2=1).real
            else:
                state = torch.flatten(
                    state,
                    start_dim=0,
                    end_dim=-2,
                ).t()
                probs = torch.abs(torch.pow(state, 2))

            probs = torch.pow(torch.abs(state), 2)
            return list(
                map(lambda p: sample_multinomial(p, self.n_qubits, n_shots), probs)
            )


class DropoutQuantumCircuit(QuantumCircuit):
    """Creates a quantum circuit able to perform quantum dropout, based on the work of https://arxiv.org/abs/2310.04120.
    Args:
        dropout_mode (DropoutMode): type of dropout to perform. Defaults to DropoutMode.ROTATIONAL
        dropout_prob (float): dropout probability. Defaults to 0.06.
    """

    def __init__(
        self,
        n_qubits: int,
        operations: list[Module],
        dropout_mode: DropoutMode = DropoutMode.ROTATIONAL,
        dropout_prob: float = 0.06,
    ):
        super().__init__(n_qubits, operations)
        self.dropout_mode = dropout_mode
        self.dropout_prob = dropout_prob

        self.dropout_fn = getattr(self, dropout_mode)

    def forward(
        self,
        state: State,
        values: dict[str, Tensor] | ParameterDict = dict(),
        embedding: Embedding | None = None,
    ) -> State:
        if self.training:
            state = self.dropout_fn(state, values)
        else:
            for op in self.operations:
                state = op(state, values)
        return state

    def rotational_dropout(
        self, state: State = None, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        """Randomly drops entangling rotational gates.

        Args:
            state (State, optional): pure state vector . Defaults to None.
            values (dict[str, Tensor] | ParameterDict, optional): gate parameters. Defaults to {}.

        Returns:
            State: pure state vector
        """
        for op in self.operations:
            if not (
                (hasattr(op, "param_name"))
                and (values[op.param_name].requires_grad)
                and not (int(1 - bernoulli(tensor(self.dropout_prob))))
            ):
                state = op(state, values)

        return state

    def entangling_dropout(
        self, state: State = None, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        """Randomly drops entangling gates.

        Args:
            state (State, optional): pure state vector. Defaults to None.
            values (dict[str, Tensor] | ParameterDict, optional): gate parameters. Defaults to {}.

        Returns:
            State: pure state vector
        """
        for op in self.operations:
            has_param = hasattr(op, "param_name")
            keep = int(1 - bernoulli(tensor(self.dropout_prob)))

            if has_param or keep:
                state = op(state, values)

        return state

    def canonical_fwd_dropout(
        self, state: State = None, values: dict[str, Tensor] | ParameterDict = dict()
    ) -> State:
        """Randomly drops rotational gates and next immediate entangling
        gates whose target bit is located on dropped rotational gates.

        Args:
            state (State, optional): pure state vector. Defaults to None.
            values (dict[str, Tensor] | ParameterDict, optional): gate parameters. Defaults to {}.

        Returns:
            State: pure state vector
        """
        entanglers_to_drop = dict.fromkeys(range(state.ndim - 1), 0)  # type: ignore
        for op in self.operations:
            if (
                hasattr(op, "param_name")
                and (values[op.param_name].requires_grad)
                and not (int(1 - bernoulli(tensor(self.dropout_prob))))
            ):
                entanglers_to_drop[op.target] = 1
            else:
                if not hasattr(op, "param_name") and (
                    entanglers_to_drop[op.control[0]] == 1
                ):
                    entanglers_to_drop[op.control[0]] = 0
                else:
                    state = op(state, values)

        return state
