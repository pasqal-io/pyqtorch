from __future__ import annotations

from typing import Any, Optional, Union

import torch

torch.set_default_dtype(torch.float64)

DEFAULT_MATRIX_DTYPE = torch.cdouble

IMAT = torch.eye(2, dtype=DEFAULT_MATRIX_DTYPE)
XMAT = torch.tensor([[0, 1], [1, 0]], dtype=DEFAULT_MATRIX_DTYPE)
YMAT = torch.tensor([[0, -1j], [1j, 0]], dtype=DEFAULT_MATRIX_DTYPE)
ZMAT = torch.tensor([[1, 0], [0, -1]], dtype=DEFAULT_MATRIX_DTYPE)
SMAT = torch.tensor([[1, 0], [0, 1j]], dtype=DEFAULT_MATRIX_DTYPE)
SDAGGERMAT = torch.tensor([[1, 0], [0, -1j]], dtype=DEFAULT_MATRIX_DTYPE)
TMAT = torch.tensor(
    [[1, 0], [0, torch.exp(torch.tensor(1.0j * torch.pi / 4))]], dtype=DEFAULT_MATRIX_DTYPE
)
NMAT = torch.tensor([[0, 0], [0, 1]], dtype=DEFAULT_MATRIX_DTYPE)
NDIAG = torch.tensor([0, 1], dtype=DEFAULT_MATRIX_DTYPE)
ZDIAG = torch.tensor([1, -1], dtype=DEFAULT_MATRIX_DTYPE)
IDIAG = torch.tensor([1, 1], dtype=DEFAULT_MATRIX_DTYPE)
SWAPMAT = torch.tensor(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=DEFAULT_MATRIX_DTYPE
)
CSWAPMAT = torch.tensor(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=DEFAULT_MATRIX_DTYPE,
)
HMAT = 1 / torch.sqrt(torch.tensor(2)) * torch.tensor([[1, 1], [1, -1]], dtype=DEFAULT_MATRIX_DTYPE)


OPERATIONS_DICT = {
    "I": IMAT,
    "X": XMAT,
    "Y": YMAT,
    "Z": ZMAT,
    "S": SMAT,
    "SDAGGER": SDAGGERMAT,
    "T": TMAT,
    "N": NMAT,
    "H": HMAT,
    "SWAP": SWAPMAT,
    "CSWAP": CSWAPMAT,
}


def _unitary(
    theta: torch.Tensor, P: torch.Tensor, I: torch.Tensor, batch_size: int  # noqa: E741
) -> torch.Tensor:
    """
    Generate a unitary parametrized by theta for a pauli operation P.
    """
    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    batch_imat = I.unsqueeze(2).repeat(1, 1, batch_size)
    batch_operation_mat = P.unsqueeze(2).repeat(1, 1, batch_size)

    return cos_t * batch_imat - 1j * sin_t * batch_operation_mat


def _dagger(matrices: torch.Tensor) -> torch.Tensor:  # noqa: E741
    """Perform the dagger operation on matrices."""
    return torch.permute(matrices.conj(), (1, 0, 2))


def _jacobian(
    theta: torch.Tensor, P: torch.Tensor, I: torch.Tensor, batch_size: int  # noqa: E741
) -> torch.Tensor:
    """
    Compute the jacobian.
    """
    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    batch_imat = I.unsqueeze(2).repeat(1, 1, batch_size)
    batch_operation_mat = P.unsqueeze(2).repeat(1, 1, batch_size)

    return -1 / 2 * (sin_t * batch_imat + 1j * cos_t * batch_operation_mat)


def make_controlled(
    matrices: torch.Tensor, batch_size: int, n_control_qubits: int = 1
) -> torch.Tensor:
    """Transform a 2x2 unitary into a controlled unitary.

    Args:

        matrices (torch.Tensor): the matrix representing the unitary which should be performed.
        batch_size (int): the batch size
        n_control_qubits (int): The number of control qubits.

    Returns:

        torch.Tensor: the resulting controlled gate populated by operation_matrix
    """
    _controlled: torch.Tensor = (
        torch.eye(2 ** (n_control_qubits + 1), dtype=DEFAULT_MATRIX_DTYPE)
        .unsqueeze(2)
        .repeat(1, 1, batch_size)
    )
    _controlled[2 ** (n_control_qubits + 1) - 2 :, 2 ** (n_control_qubits + 1) - 2 :, :] = matrices
    return _controlled


def ZZ(N: int, i: int = 0, j: int = 0, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """
    Returns the tensor representation of the ZZ interaction operator
    between qubits i and j in a quantum circuit.

    Arguments:
        N (int): The total number of qubits in the circuit.
        i (int): Index of the first qubit (default: 0).
        j (int): Index of the second qubit (default: 0).
        device (Union[str, torch.device]): Device to store the tensor on (default: "cpu").

    Returns:
        torch.Tensor: The tensor representation of the ZZ interaction operator.

    Examples:
    ```python exec="on" source="above" result="json"
    from pyqtorch.matrices import ZZ
    result=ZZ(2, 0, 1)
    print(result) #tensor([ 1.+0.j, -1.+0.j, -1.+0.j,  1.-0.j], dtype=torch.complex128)
    ```
    """
    if i == j:
        return torch.ones(2**N).to(device)

    op_list = [ZDIAG.to(device) if k in [i, j] else IDIAG.to(device) for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = torch.kron(operator, op)

    return operator


def NN(N: int, i: int = 0, j: int = 0, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    """
    Returns the tensor representation of the NN interaction operator
    between qubits i and j in a quantum circuit.

    Arguments:
        N (int): The total number of qubits in the circuit.
        i (int): Index of the first qubit (default: 0).
        j (int): Index of the second qubit (default: 0).
        device (Union[str, torch.device]): Device to store the tensor on (default: "cpu").

    Returns:
        torch.Tensor: The tensor representation of the NN interaction operator.

    Examples:
    ```python exec="on" source="above" result="json"
    from pyqtorch.matrices import NN
    result=NN(2, 0, 1)
    print(result) #tensor([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j], dtype=torch.complex128)
    ```
    """
    if i == j:
        return torch.ones(2**N, dtype=DEFAULT_MATRIX_DTYPE).to(device)

    op_list = [NDIAG.to(device) if k in [i, j] else IDIAG.to(device) for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = torch.kron(operator, op)

    return operator


def single_Z(N: int, i: int = 0, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    op_list = [ZDIAG.to(device) if k == i else IDIAG.to(device) for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = torch.kron(operator, op)

    return operator


def single_N(N: int, i: int = 0, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    op_list = [NDIAG.to(device) if k == i else IDIAG.to(device) for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = torch.kron(operator, op)

    return operator


def sum_Z(N: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    H = torch.zeros(2**N, dtype=torch.cdouble).to(device)
    for i in range(N):
        H += single_Z(N, i, device)
    return H


def sum_N(N: int, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    H = torch.zeros(2**N, dtype=torch.cdouble).to(device)
    for i in range(N):
        H += single_N(N, i, device)
    return H


def generate_ising_from_graph(
    graph: Any,  # optional library type
    precomputed_zz: Optional[torch.Tensor] = None,
    type_ising: str = "Z",
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    N = graph.number_of_nodes()
    # construct the hamiltonian
    H = torch.zeros(2**N, dtype=DEFAULT_MATRIX_DTYPE).to(device)

    for edge in graph.edges.data():
        if precomputed_zz is not None:
            if (edge[0], edge[1]) in precomputed_zz[N]:
                key = (edge[0], edge[1])
            else:
                key = (edge[1], edge[0])
            H += precomputed_zz[N][key]
        else:
            if type_ising == "Z":
                H += ZZ(N, edge[0], edge[1], device)
            elif type_ising == "N":
                H += NN(N, edge[0], edge[1], device)
            else:
                raise ValueError("'type_ising' must be in ['Z', 'N']")

    return H
