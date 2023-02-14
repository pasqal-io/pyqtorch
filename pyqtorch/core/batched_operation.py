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

import numpy as np
import torch
from numpy.typing import ArrayLike

from pyqtorch.converters.store_ops import ops_cache, store_operation
from pyqtorch.core.operation import RX, H
from pyqtorch.core.utils import _apply_batch_gate, OPERATIONS_DICT

IMAT = OPERATIONS_DICT["I"]
XMAT = OPERATIONS_DICT["X"]
YMAT = OPERATIONS_DICT["Y"]
ZMAT = OPERATIONS_DICT["Z"]


def get_parametrized_batch_for_operation(operation_type: str, theta: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """ Helper method which takes a string describing an operation type and a parameter theta vector and returns
        a batch of the corresponding parametrized rotation matrices
    Args:

    operation_type (str): the type of operation which should be performed (RX,RY,RZ)
    theta (torch.Tensor): 1D-tensor holding the values of the parameter
    batch_size (int): the batch size
    device (torch.device): the device which to run on

    Returns:
    torch.Tensor: a batch of gates after applying theta 
    """

    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    batch_imat = OPERATIONS_DICT["I"].unsqueeze(2).repeat(1, 1, batch_size).to(device)
    batch_operation_mat = OPERATIONS_DICT[operation_type].unsqueeze(2).repeat(1, 1, batch_size).to(device)

    return cos_t * batch_imat - 1j * sin_t * batch_operation_mat


def create_controlled_batch_from_operation(operation_batch: torch.Tensor, batch_size: int) -> torch.Tensor:
    """ Method which takes a 2x2 torch.Tensor and transforms it into a Controlled Operation Gate

    Args:

        operation_matrix (torch.Tensor): the type of operation which should be performed (RX,RY,RZ)
        batch_size (int): the batch size
    
    Returns:

        torch.Tensor: the resulting controlled gate populated by operation_matrix
    """
    controlled_batch: torch.Tensor = torch.eye(4, dtype=torch.cdouble).unsqueeze(2).repeat(1, 1, batch_size)
    controlled_batch[2:,2:,:] = torch.clone(operation_batch)
    return controlled_batch


def batchedRX(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
    """Parametrized single-qubit RX rotation with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

    Example:
    ```py
    import torch
    from pyqtorch.core.circuit import QuantumCircuit
    from pyqtorch.core.batched_operation import batchedRX
    
    nqubits = 4
    batch_size = 10
    
    state = QuantumCircuit(nqubits).init_state(batch_size)
    batched_params = torch.rand(batch_size)
    
    # if the length of the batched_params is not matching the batch size
    # an error will be thrown
    out_state = batchedRX(batched_params, state, [0], nqubits)
    ```

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

    batch_size = len(theta)

    mat = get_parametrized_batch_for_operation("X", theta, batch_size, dev)

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)


def batchedRY(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
    """Parametrized single-qubit RY rotation with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

    Example:
    ```py
    import torch
    from pyqtorch.core.circuit import QuantumCircuit
    from pyqtorch.core.batched_operation import batchedRY
    
    nqubits = 4
    batch_size = 10
    
    state = QuantumCircuit(nqubits).init_state(batch_size)
    batched_params = torch.rand(batch_size)
    
    # if the length of the batched_params is not matching the batch size
    # an error will be thrown
    out_state = batchedRY(batched_params, state, [0], nqubits)
    ```

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

    batch_size = len(theta)

    mat = get_parametrized_batch_for_operation("Y", theta, batch_size, dev)

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)


def batchedRZ(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
    """Parametrized single-qubit RZ rotation with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

    Example:
    ```py
    import torch
    from pyqtorch.core.circuit import QuantumCircuit
    from pyqtorch.core.batched_operation import batchedRZ
    
    nqubits = 4
    batch_size = 10
    
    state = QuantumCircuit(nqubits).init_state(batch_size)
    batched_params = torch.rand(batch_size)
    
    # if the length of the batched_params is not matching the batch size
    # an error will be thrown
    out_state = batchedRZ(batched_params, state, [0], nqubits)
    ```

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

    batch_size = len(theta)

    mat = get_parametrized_batch_for_operation("Z", theta, batch_size, dev)

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)


def batchedRZZ(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
    """Parametrized two-qubit RZZ rotation with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

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
    batch_size = len(theta)

    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((4, 4, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((4, 4, 1))

    mat = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.cdouble).to(dev))

    imat = (
        torch.eye(4, dtype=torch.cdouble).unsqueeze(2).repeat(1, 1, batch_size).to(dev)
    )
    xmat = mat.unsqueeze(2).repeat(1, 1, batch_size).to(dev)

    mat = cos_t * imat + 1j * sin_t * xmat

    return _apply_batch_gate(state, mat, qubits, N_qubits, batch_size)


def batchedRXX(
    theta: torch.Tensor, state: torch.Tensor, qubits: Any, N_qubits: int
) -> torch.Tensor:
    """Parametrized two-qubit RXX rotation with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("RXX", qubits, param=theta)

    for q in qubits:
        state = H(state, [q], N_qubits)
    state = batchedRZZ(theta, state, qubits, N_qubits)
    for q in qubits:
        state = H(state, [q], N_qubits)

    return state


def batchedRYY(
    theta: torch.Tensor, state: torch.Tensor, qubits: Any, N_qubits: int
) -> torch.Tensor:
    """Parametrized two-qubit RYY rotation with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

    Args:
        theta (torch.Tensor): 1D-tensor holding the values of the parameter
        state (torch.Tensor): the input quantum state, of shape `(N_0, N_1,..., N_N, batch_size)`
        qubits (ArrayLike): list of qubit indices where the gate will operate
        N_qubits (int): the number of qubits in the system

    Returns:
        torch.Tensor: the resulting state after applying the gate
    """

    if ops_cache.enabled:
        store_operation("RYY", qubits, param=theta)

    for q in qubits:
        state = RX(torch.tensor(np.pi / 2), state, [q], N_qubits)
    state = batchedRZZ(theta, state, qubits, N_qubits)
    for q in qubits:
        state = RX(-torch.tensor(np.pi / 2), state, [q], N_qubits)

    return state
    

def batchedCPHASE(
    theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
    """Parametrized two-qubit CPHASE gate with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

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
    batch_size = len(theta)
    mat = torch.eye(4).repeat((batch_size,1,1))
    mat = torch.permute(mat,(1,2,0))
    phase_rotation_angles = torch.exp(torch.tensor(1j) * theta).unsqueeze(0).unsqueeze(1)
    mat[3,3,:] = phase_rotation_angles

    return _apply_batch_gate(state, mat.to(dev), qubits, N_qubits, batch_size)


def batchedCRX(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
    """Parametrized two-qubit Controlled X rotation gate with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

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

    dev = state.device
    batch_size = len(theta)

    operations_batch = get_parametrized_batch_for_operation("X", theta, batch_size, dev)
    controlledX_batch = create_controlled_batch_from_operation(operations_batch, batch_size)

    return _apply_batch_gate(state, controlledX_batch, qubits, N_qubits, batch_size)



def batchedCRY(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
    """Parametrized two-qubit Controlled Y rotation gate with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

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

    dev = state.device
    batch_size = len(theta)

    operations_batch = get_parametrized_batch_for_operation("Y", theta, batch_size, dev)
    controlledX_batch = create_controlled_batch_from_operation(operations_batch, batch_size)

    return _apply_batch_gate(state, controlledX_batch, qubits, N_qubits, batch_size)


def batchedCRZ(theta: torch.Tensor, state: torch.Tensor, qubits: ArrayLike, N_qubits: int
) -> torch.Tensor:
    """Parametrized two-qubit Controlled Z rotation gate with batched parameters

    A batched operation is an operation which efficiently applies a set of parametrized
    gates with parameters held by the `theta` argument on a set of input states held by
    the `state` argument. The number of gates and input states is the batch size. For 
    large batches, this gate is much faster than its standard non-batched version 

    Notice that for this operation to work the input state must also have been 
    initialized with its *last* dimension equal to the batch size. Use the 
    QuantumCircuit.init_state() method to properly initialize a state usable 
    for batched operations

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

    dev = state.device
    batch_size = len(theta)

    operations_batch = get_parametrized_batch_for_operation("Z", theta, batch_size, dev)
    controlledX_batch = create_controlled_batch_from_operation(operations_batch, batch_size)

    return _apply_batch_gate(state, controlledX_batch, qubits, N_qubits, batch_size)