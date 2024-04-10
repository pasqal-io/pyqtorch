from __future__ import annotations

from torch.nn import Module

from pyqtorch.composite import AddOp, CompOp


class Hamiltonian(AddOp):
    def __init__(self, terms: list[list[Module]] | dict[str, list[Module]]) -> None:
        if isinstance(terms, list):
            terms = [CompOp(t) for t in terms]
        if isinstance(terms, dict):
            terms = {k: CompOp(t) for k, t in terms.items()}
        super().__init__(terms)
