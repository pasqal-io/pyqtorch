import torch
import numpy as np
from string import ascii_letters as ABC

ABC_ARRAY = np.array(list(ABC))


def apply_gate(state, mat, qubits, N_qubits):
    '''
    Apply a gate represented by its matrix `mat` to the quantum state
    `state`

    Inputs:
    - state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
    - mat (torch.Tensor): the matrix representing the gate
    - qubits (list, tuple, array): iterator containing the qubits
    the gate is applied to
    - N_qubits: the total number of qubits of the system

    Output:
    - state (torch.Tensor): the quantum state after application of the gate.
    Same shape as `ìnput_state`
    '''
    mat = mat.reshape([2] * len(qubits) * 2)
    mat_dims = list(range(len(qubits), 2 * len(qubits)))
    state_dims = [N_qubits - i - 1 for i in list(qubits)]
    axes = (mat_dims, state_dims)

    state = torch.tensordot(mat, state, dims=axes)
    inv_perm = torch.argsort(
        torch.tensor(
            state_dims + [j for j in range(N_qubits+1) if j not in state_dims]
            )
        )
    state = torch.permute(state, tuple(inv_perm))
    return state


def apply_einsum_gate(state, mat, qubits, N_qubits):
    '''
    Same as `apply_gate` but with the `torch.einsum` function

    Inputs:
    - state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
    - mat (torch.Tensor): the matrix representing the gate
    - qubits (list, tuple, array): iterator containing the qubits
    the gate is applied to
    - N_qubits: the total number of qubits of the system

    Output:
    - state (torch.Tensor): the quantum state after application of the gate.
    Same shape as `ìnput_state`
    '''
    mat = mat.reshape([2] * len(qubits) * 2)
    qubits = N_qubits - 1 - np.array(qubits)

    state_indices = ABC_ARRAY[0:N_qubits+1]
    # Create new indices for the matrix indices
    mat_indices = ABC_ARRAY[N_qubits + 2:N_qubits + 2 + 2 * len(qubits)]
    mat_indices[len(qubits):2*len(qubits)] = state_indices[qubits]

    # Create the new state indices: same as input states but
    # modified affected qubits
    new_state_indices = state_indices.copy()
    new_state_indices[qubits] = mat_indices[0:len(qubits)]

    # Transform the arrays into strings
    state_indices = "".join(list(state_indices))
    new_state_indices = "".join(list(new_state_indices))
    mat_indices = "".join(list(mat_indices))

    einsum_indices = f"{mat_indices},{state_indices}->{new_state_indices}"

    state = torch.einsum(einsum_indices, mat, state)

    return state


def apply_batch_gate(state, mat, qubits, N_qubits, batch_size):
    '''
    Apply a batch execution of gates to a circuit.
    Given an tensor of states [state_0, ... state_b] and
    an tensor of gates [G_0, ... G_b] it will return the
    tensor [G_0 * state_0, ... G_b * state_b]. All gates
    are applied to the same qubit.

    Inputs:
    - state (torch.Tensor): input state of shape [2] * N_qubits + [batch_size]
    - mat (torch.Tensor): the tensor representing the gates. The last dimension
    is the batch dimension. It has to be the sam eas the last dimention of
    `state`
    - qubits (list, tuple, array): iterator containing the qubits
    the gate is applied to
    - N_qubits (int): the total number of qubits of the system
    - batx-ch_size (int): the size of the batch

    Output:
    - state (torch.Tensor): the quantum state after application of the gate.
    Same shape as `input_state`
    '''
    mat = mat.reshape([2] * len(qubits) * 2 + [batch_size])
    qubits = N_qubits - 1 - np.array(qubits)

    state_indices = ABC_ARRAY[0:N_qubits+1].copy()
    mat_indices = ABC_ARRAY[N_qubits+2:N_qubits+2+2*len(qubits)+1].copy()
    mat_indices[len(qubits):2*len(qubits)] = state_indices[qubits]
    mat_indices[-1] = state_indices[-1]

    new_state_indices = state_indices.copy()
    new_state_indices[qubits] = mat_indices[0:len(qubits)]

    state_indices = "".join(list(state_indices))
    new_state_indices = "".join(list(new_state_indices))
    mat_indices = "".join(list(mat_indices))

    einsum_indices = f"{mat_indices},{state_indices}->{new_state_indices}"
    #print(einsum_indices)
    state = torch.einsum(einsum_indices, mat, state)

    return state
