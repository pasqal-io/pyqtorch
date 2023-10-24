from __future__ import annotations

import torch

from pyqtorch.abstract import AbstractOperator
from pyqtorch.apply import _apply_einsum

class CustomMatrixOperation(AbstractOperator):
    # Primitive or AbstractOperator, or just torch.Module ?
    # I feel like the methods will need different signatures then
    # what you have in the abstract classes.
    def __init__(
        self,
        matrix: torch.Tensor,
        qubit_support: list[int],
    ):
        # Run checks:
        # Is matrix size a power of 2?
        # I would say to allow here also non-unitary operations,
        # since it may be useful in the future for projectors (to
        # be added in qadence.)
        # Or have a separate CustomUnitary inheriting from CustomMatrixOperation

        self.matrix = matrix
        self.qubit_support = qubit_support


    def unitary(self) -> None:
        raise NotImplementedError

    def jacobian(self) -> None:
        raise NotImplementedError
    

class HamEvo(CustomMatrixOperation):
    """
    Class for Hamiltonian evolution operation, using matrix exponential method.

    Args:
        generator: Hamiltonian tensor
        t: Time parameter tensor
        qubit_support: Qubits for operation
    """

    def __init__(
        self, 
        generator: torch.Tensor, 
        t: torch.Tensor, 
        qubit_support: list[int], 
    ):
        # Going in this direction, this init is the last thing to do
        # since first we need to get the unitary matrix by exponentiating the generator
        super().__init__(matrix, qubit_support)

        if len(self.H.size()) < 3:
            self.H = self.H.unsqueeze(2)
        batch_size_h = self.H.size()[BATCH_DIM]

        # Check if all hamiltonians in the batch are diagonal
        # This would still be nice to keep, and maybe the `is_diag` function
        # can be improved. It's a big factor in speeding up HamEvo
        diag_check = torch.tensor([is_diag(self.H[..., i]) for i in range(batch_size_h)])
        self.batch_is_diag = bool(torch.prod(diag_check))

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        """
        Applies the Hamiltonian evolution operation
        on the given state using matrix exponential method.

        Args:
            state (tensor): Input quantum state.

        Returns:
            tensor: Output state after Hamiltonian evolution.
        """

        batch_size_t = len(self.t)
        batch_size_h = self.H.size()[BATCH_DIM]
        t_evo = self.t

        if self.batch_is_diag:
            # Skips the matrix exponential for diagonal hamiltonians
            H_diagonals = torch.diagonal(self.H)
            evol_exp_arg = H_diagonals * (-1j * t_evo).view((-1, 1))
            evol_operator_T = torch.diag_embed(torch.exp(evol_exp_arg))
            evol_operator = torch.transpose(evol_operator_T, 0, -1)
        else:
            H_T = torch.transpose(self.H, 0, -1)
            evol_exp_arg = H_T * (-1j * t_evo).view((-1, 1, 1))
            evol_operator_T = torch.linalg.matrix_exp(evol_exp_arg)
            evol_operator = torch.transpose(evol_operator_T, 0, -1)

        # Basically this evol_operator is the final unitary matrix to get

        batch_size = max(batch_size_h, batch_size_t)
        return _apply_einsum(state, evol_operator, self.qubit_support, self.n_qubits, batch_size)