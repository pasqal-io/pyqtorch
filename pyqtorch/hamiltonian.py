from __future__ import annotations

from torch import Tensor
from torch.nn import Module

from pyqtorch.composite import AddOps, SeqOps


class Hamiltonian(AddOps):
    def __init__(self, terms: list[list[Module]] | dict[str, list[Module]]) -> None:
        if isinstance(terms, list):
            terms = [SeqOps(term) for term in terms]
        if isinstance(terms, dict):
            terms = {key: SeqOps(term) for key, term in terms.items()}
        super().__init__(terms)


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
