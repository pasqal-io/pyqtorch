# Copyright 2022 PyQ Development Team

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional, Tuple

import torch
from numpy.typing import ArrayLike

from pyqtorch.converters.store_ops import ops_cache, store_operation
from pyqtorch.core.utils import OPERATIONS_DICT, _apply_gate


def get_parametrized_matrix_for_operation(operation_type: str, theta: torch.Tensor) -> torch.Tensor:
    """Helper method which takes a string describing an operation type and a
    parameter theta and returns the corresponding parametrized rotation matrix

    Args:
        operation_type (str): the type of operation which should be performed (RX,RY,RZ)
        theta (torch.Tensor): 1D-tensor holding the values of the parameter

    Returns:
        torch.Tensor: the resulting gate after applying theta
    """
    return OPERATIONS_DICT["I"] * torch.cos(theta / 2) - 1j * OPERATIONS_DICT[
        operation_type
    ] * torch.sin(theta / 2)


def create_controlled_matrix_from_operation(
    operation_matrix: torch.Tensor, n_control_qubits: int = 1
) -> torch.Tensor:
    """Method which takes a torch.Tensor and transforms it into a Controlled Operation Gate

    Args:

        operation_matrix: (torch.Tensor): the type of operation which should be
        performed (RX,RY,RZ,SWAP)
        n_control_qubits: (int): The number of control qubits used

    Returns:

        torch.Tensor: the resulting controlled gate populated by operation_matrix
    """
    mat_size = len(operation_matrix)
    controlled_mat: torch.Tensor = torch.eye(2**n_control_qubits * mat_size, dtype=torch.cdouble)
    controlled_mat[-mat_size:, -mat_size:] = operation_matrix
    return controlled_mat


def RX(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Parametrized single-qubit RX rotation

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """
    if ops_cache.enabled:
        store_operation("RX", qubits, param=theta)

    dev = state.device
    mat: torch.Tensor = get_parametrized_matrix_for_operation("X", theta).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def RY(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Parametrized single-qubit RY rotation

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """
    if ops_cache.enabled:
        store_operation("RY", qubits, param=theta)

    dev = state.device
    mat: torch.Tensor = get_parametrized_matrix_for_operation("Y", theta).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def RZ(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Parametrized single-qubit RZ rotation

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """
    if ops_cache.enabled:
        store_operation("RZ", qubits, param=theta)

    dev = state.device
    mat: torch.Tensor = get_parametrized_matrix_for_operation("Z", theta).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def RZZ(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Parametrized two-qubits RZ rotation

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """
    if ops_cache.enabled:
        store_operation("RZZ", qubits, param=theta)

    dev = state.device
    mat = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.cdouble).to(dev))
    mat = 1j * torch.sin(theta / 2) * mat + torch.cos(theta / 2) * torch.eye(
        4, dtype=torch.cdouble
    ).to(dev)
    return _apply_gate(state, torch.diag(mat), qubits, N_qubits)


def U(
    phi: torch.Tensor,
    theta: torch.Tensor,
    omega: torch.Tensor,
    state: torch.Tensor,
    qubits: ArrayLike,
    N_qubits: int,
) -> torch.Tensor:
    """Parametrized arbitrary rotation along the axes of the Bloch sphere

    The angles `phi, theta, omega` in tensor format, applied as:
    U(phi, theta, omega) = RZ(omega)RY(theta)RZ(phi)

    Args:
        phi (torch.Tensor): 1D-tensor holding the values of the `phi` parameter
        theta (torch.Tensor): 1D-tensor holding the values of the `theta` parameter
        omega (torch.Tensor): 1D-tensor holding the values of the `omega` parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("U", qubits, param=[phi, theta, omega])  # type: ignore[list-item]

    dev = state.device
    t_plus = torch.exp(-1j * (phi + omega) / 2)
    t_minus = torch.exp(-1j * (phi - omega) / 2)
    mat = (
        torch.tensor([[1, 0], [0, 0]], dtype=torch.cdouble).to(dev) * torch.cos(theta / 2) * t_plus
        - torch.tensor([[0, 1], [0, 0]], dtype=torch.cdouble).to(dev)
        * torch.sin(theta / 2)
        * torch.conj(t_minus)
        + torch.tensor([[0, 0], [1, 0]], dtype=torch.cdouble).to(dev)
        * torch.sin(theta / 2)
        * t_minus
        + torch.tensor([[0, 0], [0, 1]], dtype=torch.cdouble).to(dev)
        * torch.cos(theta / 2)
        * torch.conj(t_plus)
    )
    return _apply_gate(state, mat, qubits, N_qubits)


def I(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:  # noqa: E743
    """I single-qubit gate

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("I", qubits)

    dev = state.device
    mat = OPERATIONS_DICT["I"].to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def X(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """X single-qubit gate

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("X", qubits)

    dev = state.device
    mat = OPERATIONS_DICT["X"].to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def Z(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Z single-qubit gate

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("Z", qubits)

    dev = state.device
    mat = OPERATIONS_DICT["Z"].to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def Y(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Y single-qubit gate

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("Y", qubits)

    dev = state.device
    mat = OPERATIONS_DICT["Y"].to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def H(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Hadamard single-qubit gate

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("H", qubits)

    dev = state.device
    mat = OPERATIONS_DICT["H"].to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def ControlledOperationGate(
    state: torch.Tensor,
    qubits: ArrayLike,
    N_qubits: int,
    operation_matrix: torch.Tensor,
) -> torch.Tensor:
    """Generalized Controlled Rotation gate with two-qubits support

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system
        operation_matrix (torch.Tensor): a tensor holding the parameters for the
            operation (RX,RY,RZ)

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """
    dev = state.device
    controlled_operation_matrix: torch.Tensor = create_controlled_matrix_from_operation(
        operation_matrix
    )
    return _apply_gate(state, controlled_operation_matrix.to(dev), qubits, N_qubits)


def CNOT(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Controlled NOT gate with two-qubits support

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:

        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("CNOT", qubits)

    return ControlledOperationGate(state, qubits, N_qubits, OPERATIONS_DICT["X"])


def CRX(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Controlled RX rotation gate with two-qubits support

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """
    if ops_cache.enabled:
        store_operation("CRX", qubits, param=theta)

    operation_matrix: torch.Tensor = get_parametrized_matrix_for_operation("X", theta)
    return ControlledOperationGate(state, qubits, N_qubits, operation_matrix)


def CRY(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Controlled RY rotation gate with two-qubits support

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("CRY", qubits, param=theta)

    operation_matrix: torch.Tensor = get_parametrized_matrix_for_operation("Y", theta)
    return ControlledOperationGate(state, qubits, N_qubits, operation_matrix)


def CRZ(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Controlled RZ rotation gate with two-qubits support

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("CRZ", qubits, param=theta)

    operation_matrix: torch.Tensor = get_parametrized_matrix_for_operation("Z", theta)
    return ControlledOperationGate(state, qubits, N_qubits, operation_matrix)


def S(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """S single-qubit gate

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("S", qubits)

    dev = state.device
    mat = OPERATIONS_DICT["S"].to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def Sdagger(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Sdagger single-qubit gate

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("Sdagger", qubits)

    dev = state.device
    mat = OPERATIONS_DICT["SDAGGER"].to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def T(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """T single-qubit gate

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("T", qubits)

    dev = state.device
    mat = OPERATIONS_DICT["T"].to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def SWAP(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """SWAP 2-qubit gate

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("SWAP", qubits)

    dev = state.device
    mat = OPERATIONS_DICT["SWAP"].to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def CPHASE(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
    """Parametrized 2-qubit CPHASE gate

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """
    if ops_cache.enabled:
        store_operation("CPHASE", qubits, param=theta)

    dev = state.device
    mat: torch.Tensor = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, torch.exp(torch.tensor(1j * theta))],
        ],
        dtype=torch.cdouble,
    ).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def CSWAP(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Controlled SWAP gate with three-qubit support

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:

        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("CSWAP", qubits)

    return ControlledOperationGate(state, qubits, N_qubits, OPERATIONS_DICT["SWAP"])


def hamiltonian_evolution(
    H: torch.Tensor,
    state: torch.Tensor,
    t: torch.Tensor,
    qubits: Any,
    N_qubits: int,
    n_steps: int = 100,
) -> torch.Tensor:
    """A function to perform time-evolution according to the generator `H` acting on a
    `N_qubits`-sized input `state`, for a duration `t`. See also tutorials for more information
    on how to use this gate.

    Args:
        H (torch.Tensor): the dense matrix representing the Hamiltonian,
            provided as a `Tensor` object with shape
            `(N_0,N_1,...N_(N**2),batch_size)`, i.e. the matrix is reshaped into
            the list of its rows
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        t (torch.Tensor): the evolution time, real for default unitary evolution
        qubits (Any): The qubits support where the H evolution is applied
        N_qubits (int): The number of qubits
        n_steps (int, optional): The number of steps to divide the time interval
            in. Defaults to 100.

    Returns:
        torch.Tensor: replaces state with the evolved state according to the
            instructions above (save a copy of `state` if you need further
            processing on it)
    """
    _state = state.clone()
    if ops_cache.enabled:
        store_operation("hevo", qubits, param=t)

    h = t.reshape((1, -1)) / n_steps
    for _ in range(N_qubits - 1):
        h = h.unsqueeze(0)

    h = h.expand_as(_state)

    for _ in range(n_steps):
        k1 = -1j * _apply_gate(_state, H, qubits, N_qubits)
        k2 = -1j * _apply_gate(_state + h / 2 * k1, H, qubits, N_qubits)
        k3 = -1j * _apply_gate(_state + h / 2 * k2, H, qubits, N_qubits)
        k4 = -1j * _apply_gate(_state + h * k3, H, qubits, N_qubits)
        _state += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return _state


@lru_cache(maxsize=256)
def diagonalize(H: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Diagonalizes an Hermitian Hamiltonian, returning eigenvalues and eigenvectors.
    First checks if it's already diagonal, and second checks if H is real.
    """

    def is_diag(H: torch.Tensor) -> bool:
        return len(torch.abs(torch.triu(H, diagonal=1)).to_sparse().coalesce().values()) == 0

    def is_real(H: torch.Tensor) -> bool:
        return len(torch.imag(H).to_sparse().coalesce().values()) == 0

    if is_diag(H):
        # Skips diagonalization
        eig_values = torch.diagonal(H)
        eig_vectors = None
    else:
        if is_real(H):
            eig_values, eig_vectors = torch.linalg.eigh(H.real)
            eig_values = eig_values.to(torch.cdouble)
            eig_vectors = eig_vectors.to(torch.cdouble)
        else:
            eig_values, eig_vectors = torch.linalg.eigh(H)

    return eig_values, eig_vectors


def hamiltonian_evolution_eig(
    H: torch.Tensor,
    state: torch.Tensor,
    t: torch.Tensor,
    qubits: Any,
    N_qubits: int,
) -> torch.Tensor:
    """A function to perform time-evolution according to the generator `H` acting on a
    `N_qubits`-sized input `state`, for a duration `t`. See also tutorials for more information
    on how to use this gate. Uses exact diagonalization.

    Args:
        H (torch.Tensor): the dense matrix representing the Hamiltonian,
            provided as a `Tensor` object with shape
            `(N_0,N_1,...N_(N**2),batch_size)`, i.e. the matrix is reshaped into
            the list of its rows
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        t (torch.Tensor): the evolution time, real for default unitary evolution
        qubits (Any): The qubits support where the H evolution is applied
        N_qubits (int): The number of qubits

    Returns:
        torch.Tensor: replaces state with the evolved state according to the
            instructions above (save a copy of `state` if you need further processing on it)
    """
    _state = state.clone()
    batch_size_s = state.size()[-1]
    batch_size_t = len(t)

    t_evo = torch.zeros(batch_size_s).to(torch.cdouble)

    if batch_size_t >= batch_size_s:
        t_evo = t[:batch_size_s]
    else:
        if batch_size_t == 1:
            t_evo[:] = t[0]
        else:
            t_evo[:batch_size_t] = t

    if ops_cache.enabled:
        store_operation("hevo", qubits, param=t)

    eig_values, eig_vectors = diagonalize(H)

    if eig_vectors is None:
        for i, t_val in enumerate(t_evo):
            # Compute e^(-i H t)
            evol_operator = torch.diag(torch.exp(-1j * eig_values * t_val))
            _state[..., [i]] = _apply_gate(state[..., [i]], evol_operator, qubits, N_qubits)

    else:
        for i, t_val in enumerate(t_evo):
            # Compute e^(-i D t)
            eig_exp = torch.diag(torch.exp(-1j * eig_values * t_val))
            # e^(-i H t) = V.e^(-i D t).V^\dagger
            evol_operator = torch.matmul(
                torch.matmul(eig_vectors, eig_exp),
                torch.conj(eig_vectors.transpose(0, 1)),
            )
            _state[..., [i]] = _apply_gate(state[..., [i]], evol_operator, qubits, N_qubits)

    return _state
