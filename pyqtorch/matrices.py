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

from typing import Any, Optional, Union

import torch

IMAT = torch.tensor([1, 1], dtype=torch.cdouble)
ZMAT = torch.tensor([1, -1], dtype=torch.cdouble)
NMAT = torch.tensor([0, 1], dtype=torch.cdouble)


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

    op_list = [ZMAT.to(device) if k in [i, j] else IMAT.to(device) for k in range(N)]
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
        return torch.ones(2**N, dtype=torch.cdouble).to(device)

    op_list = [NMAT.to(device) if k in [i, j] else IMAT.to(device) for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = torch.kron(operator, op)

    return operator


def single_Z(N: int, i: int = 0, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    op_list = [ZMAT.to(device) if k == i else IMAT.to(device) for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = torch.kron(operator, op)

    return operator


def single_N(N: int, i: int = 0, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    op_list = [NMAT.to(device) if k == i else IMAT.to(device) for k in range(N)]
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
    H = torch.zeros(2**N, dtype=torch.cdouble).to(device)

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
