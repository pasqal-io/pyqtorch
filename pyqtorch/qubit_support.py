from __future__ import annotations
from functools import cached_property
from pyqtorch.utils import (
    qubit_support_as_tuple,
)

class Support:
    """
    Generic representation of the qubit support. For single qubit operations,
    a multiple index support indicates apply the operation for each index in the
    support.

    Both target and control lists must be ordered!

    Attributes:
       target = Index or indices where the operation is applied.
       control = Index or indices to which the operation is conditioned to.
    """

    def __init__(
        self,
        target: int | tuple[int, ...],
        control: int | tuple[int, ...] | None = None,
    ) -> None:
        self.target = qubit_support_as_tuple(target)
        self.control = qubit_support_as_tuple(control) if control is not None else ()
        # if self.qubits != tuple(set(self.qubits)):
        #    raise ValueError("One or more qubits are defined both as control and target.")

    @classmethod
    def target_all(cls) -> Support:
        return Support(target=())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Support):
            return NotImplemented

        return self.target == other.target and self.control == other.control

    def __len__(self):
        return len(self.qubits)

    @cached_property
    def qubits(self) -> tuple[int, ...]:
        return self.control + self.target

    @cached_property
    def sorted_qubits(self) -> tuple[int, ...]:
        return tuple(sorted(self.qubits))

    def __repr__(self) -> str:
        if not self.target:
            return f"{self.__class__.__name__}.target_all()"

        subspace = f"target: {self.target}"
        if self.control:
            subspace += f", control: {self.control}"

        return f"{self.__class__.__name__}({subspace})"