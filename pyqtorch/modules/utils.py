from __future__ import annotations

import torch


def is_diag(H: torch.Tensor) -> bool:
    """
        Check if the Hamiltonian H is diagonal.

        Args:
            H (torch.Tensor): The Hamiltonian tensor.

        Returns:
            bool: True if the Hamiltonian is diagonal, False otherwise.

        Examples:
            ```python exec="on" source="above" result="json"
            import torch
            from pyqtorch.utils import is_diag

            H = torch.tensor([[1, 0], [0, 2]])
            print(is_diag(H))  # True

            H = torch.tensor([[1, 2], [3, 4]])
            print(is_diag(H))  # False
            ```
        """
    return len(torch.abs(torch.triu(H, diagonal=1)).to_sparse().coalesce().values()) == 0


def is_real(H: torch.Tensor) -> bool:
    """
       Check if the Hamiltonian H is real.

       Args:
           H (torch.Tensor): The Hamiltonian tensor.

       Returns:
           bool: True if the Hamiltonian is real, False otherwise.

       Examples:
           ```python exec="on" source="above" result="json"
           import torch
           from pyqtorch.utils import is_real

           H = torch.tensor([[1, 0], [0, 2]])
           print(is_real(H))  # True

           H = torch.tensor([[1, 2j], [-3j, 4]])
           print(is_real(H))  # False
           ```
       """
    return len(torch.imag(H).to_sparse().coalesce().values()) == 0
