from __future__ import annotations

from torch.nn import Module

from pyqtorch.composite import AddOps, SeqOps


class Hamiltonian(AddOps):
    def __init__(self, terms: list[list[Module]] | dict[str, list[Module]]) -> None:
        if isinstance(terms, list):
            terms = [SeqOps(term) for term in terms]
        if isinstance(terms, dict):
            terms = {key: SeqOps(term) for key, term in terms.items()}
        super().__init__(terms)
