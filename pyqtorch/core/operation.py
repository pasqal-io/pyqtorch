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

from typing import Any

import torch
from numpy.typing import ArrayLike

from pyqtorch.converters.store_ops import ops_cache, store_operation
from pyqtorch.core.utils import _apply_gate

IMAT = torch.eye(2, dtype=torch.cdouble)
XMAT = torch.tensor([[0, 1], [1, 0]], dtype=torch.cdouble)
YMAT = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cdouble)
ZMAT = torch.tensor([[1, 0], [0, -1]], dtype=torch.cdouble)
SMAT = torch.tensor([[1, 0], [0, 1j]], dtype=torch.cdouble)
TMAT = torch.tensor([[1, 0], [0, torch.exp(torch.tensor(1j) * torch.pi / 4)]], dtype=torch.cdouble)
SWAPMAT = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.cdouble)


def RX(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
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
    mat: torch.Tensor = IMAT.to(dev) * torch.cos(theta / 2) - 1j * XMAT.to(
        dev
    ) * torch.sin(theta / 2)
    return _apply_gate(state, mat, qubits, N_qubits)


def RY(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
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
    mat = IMAT.to(dev) * torch.cos(theta / 2) - 1j * YMAT.to(dev) * torch.sin(theta / 2)
    return _apply_gate(state, mat, qubits, N_qubits)


def RZ(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
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
    mat = IMAT.to(dev) * torch.cos(theta / 2) + 1j * ZMAT.to(dev) * torch.sin(theta / 2)
    return _apply_gate(state, mat, qubits, N_qubits)


def RZZ(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
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
        torch.tensor([[1, 0], [0, 0]], dtype=torch.cdouble).to(dev)
        * torch.cos(theta / 2)
        * t_plus
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
    mat = XMAT.to(dev)
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
    mat = ZMAT.to(dev)
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
    mat = YMAT.to(dev)
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
    mat = (
        1
        / torch.sqrt(torch.tensor(2))
        * torch.tensor([[1, 1], [1, -1]], dtype=torch.cdouble).to(dev)
    )
    return _apply_gate(state, mat, qubits, N_qubits)


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

    dev = state.device
    mat = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.cdouble
    ).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


def CRZ(state: torch.Tensor, qubits: ArrayLike, N_qubits: int) -> torch.Tensor:
    """Controlled RZ gate with two-qubits support

    Args:
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """
    if ops_cache.enabled:
        store_operation("CRZ", qubits)

    dev = state.device
    mat = torch.tensor(
        [[1, 0, 0, 0], [0, torch.exp((-1j * torch.tensor(theta)) / 2), 0, 0], [0, 0, 1, 0], [0, 0, 0, torch.exp((1j * torch.tensor(theta)) / 2)]], dtype=torch.cdouble
    ).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)
    

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
    mat = SMAT.to(dev)
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
    mat = TMAT.to(dev)
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
    mat = SWAPMAT.to(dev)
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
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, torch.exp(torch.tensor(1j * theta))]], dtype=torch.cdouble
    ).to(dev)
    return _apply_gate(state, mat, qubits, N_qubits)


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
        H (torch.Tensor): the dense matrix representing the Hamiltonian, provided as a `Tensor` object with 
        shape  `(N_0,N_1,...N_(N**2),batch_size)`, i.e. the matrix is reshaped into the list of its rows
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        t (torch.Tensor): the evolution time, real for default unitary evolution
        qubits (Any): The qubits support where the H evolution is applied
        N_qubits (int): The number of qubits
        n_steps (int, optional): The number of steps to divide the time interval in. Defaults to 100.

    Returns:
        torch.Tensor: replaces state with the evolved state according to the instructions above (save a copy of `state`
        if you need further processing on it)
    """
    
    if ops_cache.enabled:
        store_operation("hevo", qubits, param=t)

    # batch_size = len(t)
    # #permutation = [N_qubits - q - 1 for q in qubits]
    # permutation = [N_qubits - q - 1 for q in range(N_qubits) if q not in qubits]
    # permutation += [N_qubits - q - 1 for q in qubits]
    # permutation.append(N_qubits)
    # inverse_permutation = list(np.argsort(permutation))

    # new_dim = [2] * (N_qubits - len(qubits)) + [batch_size] + [2**len(qubits)]
    # state = state.permute(*permutation).reshape((2**len(qubits), -1))

    h = t.reshape((1, -1)) / n_steps
    for _ in range(N_qubits - 1):
        h = h.unsqueeze(0)

    h = h.expand_as(state)
    # h = h.expand(2**len(qubits), -1, 2**(N_qubits - len(qubits))).reshape((2**len(qubits), -1))
    for _ in range(n_steps):
        k1 = -1j * _apply_gate(state, H, qubits, N_qubits)
        k2 = -1j * _apply_gate(state + h / 2 * k1, H, qubits, N_qubits)
        k3 = -1j * _apply_gate(state + h / 2 * k2, H, qubits, N_qubits)
        k4 = -1j * _apply_gate(state + h * k3, H, qubits, N_qubits)
        # print(state.shape)
        # k1 = -1j * torch.matmul(H, state)
        # k2 = -1j * torch.matmul(H, state + h/2 * k1)
        # k3 = -1j * torch.matmul(H, state + h/2 * k2)
        # k4 = -1j * torch.matmul(H, state + h * k3)

        state += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return state  # .reshape([2]*N_qubits + [batch_size]).permute(*inverse_permutation)
