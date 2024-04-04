from __future__ import annotations

from functools import reduce
from operator import add

from torch import Tensor
from torch.nn import Module, ModuleDict, ModuleList, ParameterDict

from pyqtorch.utils import State


def run_term(state: State, term: ModuleList) -> State:
    for op in term:
        state = op(state)
    return state


class Hamiltonian(Module):
    def __init__(self, terms: list[list[Module]] | dict[str, list[Module]]) -> None:
        super().__init__()
        if isinstance(terms, list):
            self.terms = ModuleList([ModuleList(term) for term in terms])
            self.is_parameterized = False
        if isinstance(terms, dict):
            self.terms = ModuleDict({param: ModuleList(term) for param, term in terms.items()})
            self.is_parameterized = True

    def forward(self, state: State, values: dict[str, Tensor] | ParameterDict = {}) -> State:
        if self.is_parameterized:
            return reduce(
                add, [values[param] * run_term(state, term) for param, term in self.terms.items()]
            )
        else:
            return reduce(add, [run_term(state, term) for term in self.terms])


if __name__ == "__main__":
    from pyqtorch.primitive import Z
    from pyqtorch.utils import inner_prod, product_state

    state = product_state("00")

    # Total magnetization
    n_qubits = 2
    terms_fixed = [[Z(i)] for i in range(n_qubits)]

    # Parameterized total magnetization
    terms_param = {f"z_{i}": [Z(i)] for i in range(n_qubits)}

    observable = Hamiltonian(terms_param)

    values = {f"z_{i}": Tensor([1.0]) for i in range(n_qubits)}

    exp_val = inner_prod(state, observable(state, values)).real

    print(exp_val)
