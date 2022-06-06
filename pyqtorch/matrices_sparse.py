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

import torch
import scipy.sparse as sp
import numpy as np

from itertools import combinations

IMAT = sp.coo_matrix(np.eye(2, dtype=np.cdouble))
XMAT = sp.coo_matrix(np.array([[0, 1], [1, 0]], dtype=np.cdouble))
YMAT = sp.coo_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.cdouble))
ZMAT = sp.coo_matrix(np.array([[1, 0], [0, -1]], dtype=np.cdouble))
NMAT = sp.coo_matrix(np.array([0, 1], dtype=np.cdouble))

def XX(N, i=0, j=0, device='cpu'):
    op_list = [XMAT.copy() if k in [i, j]
               else IMAT.copy() for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = sp.kron(operator, op, format='coo')
    return operator
    
def YY(N, i=0, j=0, device='cpu'):
    op_list = [YMAT.copy() if k in [i, j]
               else IMAT.copy() for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = sp.kron(operator, op, format='coo')
    return operator
    
def ZZ(N, i=0, j=0, device='cpu'):
    op_list = [ZMAT.copy() if k in [i, j]
               else IMAT.copy() for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = sp.kron(operator, op, format='coo')
    return operator


def NN(N, i=0, j=0, device='cpu'):
    op_list = [NMAT.copy() if k in [i, j]
               else IMAT.copy() for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = sp.kron(operator, op, format='coo')
    return operator


def single_Z(N, i=0, device='cpu'):
    op_list = [ZMAT.copy() if k == i
               else IMAT.copy() for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = sp.kron(operator, op, format='coo')
    return operator


def single_N(N, i=0, device='cpu'):
    op_list = [NMAT.copy() if k == i
               else IMAT.copy() for k in range(N)]
    operator = op_list[0]
    for op in op_list[1::]:
        operator = sp.kron(operator, op, format='coo')
    return operator


def sum_Z(N, device='cpu'):
    H = torch.zeros(2**N, dtype=torch.cdouble).to(device)
    for i in range(N):
        H += single_Z(N, i, device)
    return H


def sum_N(N, device='cpu'):
    H = torch.zeros(2**N, dtype=torch.cdouble).to(device)
    for i in range(N):
        H += single_N(N, i, device)
    return H


def generate_ising_from_graph(graph,
                              precomputed_zz=None,
                              type_ising='Z',
                              device='cpu'):
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
            if type_ising == 'Z':
                H += ZZ(N, edge[0], edge[1], device).copy()
            elif type_ising == 'N':
                H += NN(N, edge[0], edge[1], device).copy()
            else:
                raise ValueError("'type_ising' must be in ['Z', 'N']")

    return H


def general_hamiltonian(graph=None, alpha=None, beta=None, gamma=None, device='cpu'):
    #alpha, beta, gamma: matrices of parameters (e.g. alpha_ij)
    #connectivity_graph: which qubits are connected
    #no 1 body terms
    N = alpha.shape[0]
    # construct the hamiltonian
    H = sp.coo_matrix((2**N, 2**N))
    for edge in combinations(range(N), 2):
        if alpha is not None:
            if alpha[edge[0], edge[1]] > 1e-15:
                h = alpha[edge[0], edge[1]] * XX(N, edge[0], edge[1]).copy()
                H +=  h.copy()#alpha_ij * XX
        if beta is not None:
            if beta[edge[0], edge[1]] > 1e-15:
                h = beta[edge[0], edge[1]] * YY(N, edge[0], edge[1]).copy()
                H += h.copy() #beta_ij * YY
        if gamma is not None:
            if gamma[edge[0], edge[1]] > 1e-15:
                h = gamma[edge[0], edge[1]] * ZZ(N, edge[0], edge[1]).copy()
                H += h.copy() #gamma_ij * ZZ
    return H


def get_sparse_torch(coo_matrix):
    values = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_matrix.shape

    tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape), dtype=torch.cdouble)
    return tensor

def heisenberg_hamiltonian(graph, alpha, beta, gamma, device='cpu'):
    #to do
    pass

def XY_hamiltonian(graph, alpha, beta, gamma, device='cpu'):
    #to do
    
    return H