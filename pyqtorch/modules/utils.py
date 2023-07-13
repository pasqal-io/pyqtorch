from __future__ import annotations

import torch


def normalize(wf: torch.Tensor) -> torch.Tensor:
    return wf / torch.sqrt((wf.abs() ** 2).sum())


def is_normalized(state: torch.Tensor, atol: float = 1e-15) -> bool:
    n_qubits = len(state.size()) - 1
    batch_size = state.size()[-1]
    state = state.reshape((2**n_qubits, batch_size))
    sum_probs = (state.abs() ** 2).sum(dim=0)
    ones = torch.ones(batch_size)
    return torch.allclose(sum_probs, ones, rtol=0.0, atol=atol)  # type:ignore[no-any-return]


def is_diag(H: torch.Tensor) -> bool:
    """
    Returns True if Hamiltonian H is diagonal.
    """
    return len(torch.abs(torch.triu(H, diagonal=1)).to_sparse().coalesce().values()) == 0


def is_real(H: torch.Tensor) -> bool:
    """
    Returns True if Hamiltonian H is real.
    """
    return len(torch.imag(H).to_sparse().coalesce().values()) == 0


def overlap(state: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    n_qubits = len(state.size()) - 1
    batch_size = state.size()[-1]
    state = state.reshape((2**n_qubits, batch_size))
    other = other.reshape((2**n_qubits, batch_size))
    res = []
    for i in range(batch_size):
        ovrlp = torch.real(torch.sum(torch.conj(state[:, i]) * other[:, i]))
        res.append(ovrlp)
    return torch.stack(res)


def rot_matrices(
    theta: torch.Tensor, P: torch.Tensor, I: torch.Tensor, batch_size: int  # noqa: E741
) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: a batch of gates after applying theta
    """
    cos_t = torch.cos(theta / 2).unsqueeze(0).unsqueeze(1)
    cos_t = cos_t.repeat((2, 2, 1))
    sin_t = torch.sin(theta / 2).unsqueeze(0).unsqueeze(1)
    sin_t = sin_t.repeat((2, 2, 1))

    batch_imat = I.unsqueeze(2).repeat(1, 1, batch_size)
    batch_operation_mat = P.unsqueeze(2).repeat(1, 1, batch_size)

    return cos_t * batch_imat - 1j * sin_t * batch_operation_mat


def zero_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    """
    Generates the zero state for a specified number of qubits.

    Arguments:
        n_qubits (int): The number of qubits for which the zero state is to be generated.
        batch_size (int): The batch size for the zero state.
        device (str): The device on which the zero state tensor is to be allocated eg cpu or gpu.
        dtype (torch.cdouble): The data type of the zero state tensor.

    Returns:
        torch.Tensor: A tensor representing the zero state.
        The shape of the tensor is (batch_size, 2^n_qubits),
        where 2^n_qubits is the total number of possible states for the given number of qubits.
        The data type of the tensor is specified by the dtype parameter.

    Examples:
    ```python exec="on" source="above" result="json"
    import torch
    import pyqtorch.modules as pyq

    state = pyq.zero_state(n_qubits=2)
    print(state)  #tensor([[[1.+0.j],[0.+0.j]],[[0.+0.j],[0.+0.j]]], dtype=torch.complex128)
    ```
    """
    state = torch.zeros((2**n_qubits, batch_size), dtype=dtype, device=device)
    state[0] = 1
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


def uniform_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    """
    Generates the uniform state for a specified number of qubits.
    Returns a tensor representing the uniform state.
    The shape of the tensor is (2^n_qubits, batch_size),
    where 2^n_qubits is the total number of possible states for the given number of qubits.
    The data type of the tensor is specified by the dtype parameter.
    Each element of the tensor is initialized to 1/sqrt(2^n_qubits),
    ensuring that the total probability of the state is equal to 1.

    Arguments:
        n_qubits (int): The number of qubits for which the uniform state is to be generated.
        batch_size (int): The batch size for the uniform state.
        device (str): The device on which the uniform state tensor is to be allocated.
        dtype (torch.cdouble): The data type of the uniform state tensor.

    Returns:
        torch.Tensor: A tensor representing the uniform state.


    Examples:
    ```python exec="on" source="above" result="json"
    import torch
    import pyqtorch.modules as pyq

    state = pyq.uniform_state(n_qubits=2)
    print(state)
    #tensor([[[0.5000+0.j],[0.5000+0.j]],[[0.5000+0.j],[0.5000+0.j]]], dtype=torch.complex128)
    ```
    """
    state = torch.ones((2**n_qubits, batch_size), dtype=dtype, device=device)
    state = state / torch.sqrt(torch.tensor(2**n_qubits))
    state = state.reshape([2] * n_qubits + [batch_size])
    return state


def random_state(
    n_qubits: int,
    batch_size: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.cdouble,
) -> torch.Tensor:
    def _rand(n_qubits: int) -> torch.Tensor:
        N = 2**n_qubits
        x = -torch.log(torch.rand(N))
        sumx = torch.sum(x)
        phases = torch.rand(N) * 2.0 * torch.pi
        return normalize(
            (torch.sqrt(x / sumx) * torch.exp(1j * phases)).reshape(N, 1).type(dtype).to(device)
        )

    _state = torch.concat(tuple(_rand(n_qubits) for _ in range(batch_size)), dim=1)
    return _state.reshape([2] * n_qubits + [batch_size])
