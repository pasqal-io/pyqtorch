# Hamiltonian Evolution Module Documentation

## Overview

The Hamiltonian Evolution (`HamiltonianEvolution`) module is designed for performing quantum operations using different Hamiltonian evolution strategies, such as 4th order Runge-Kutta (RK4), Eigenvalue Decomposition, and Matrix Exponential.

This module also features a function `diagonalize()` that performs an eigenvalue decomposition on a given Hamiltonian. This function checks if the input Hamiltonian is already diagonal and real before computing the decomposition, thus saving computational resources when the checks are met.

## Class Definitions

### `HamiltonianEvolution`

This class is a PyTorch module designed to encapsulate Hamiltonian Evolution operations. The evolution operation performed by the module depends on the strategy specified through the `hamevo_type` attribute, which can be a member of the `HamEvoType` enumeration or an equivalent string ("RK4", "EIG", "EXP"). Default is set to HamEvoEXP.

### `HamEvo`

This class is the base class for Hamiltonian evolution operations, which performs the evolution operation using the Runge-Kutta 4 (RK4) numerical method. It provides `apply()` method that takes a quantum state tensor as an input and applies the Hamiltonian evolution operation. The `forward()` method provides a simplified interface for applying the operation.

### `HamEvoEig`

A subclass of `HamEvo`, this class performs the Hamiltonian evolution operation using the Eigenvalue Decomposition method. In addition to performing the eigenvalue decomposition, it also provides checks if the Hamiltonian is already diagonal, and if so, skips the decomposition computation.

### `HamEvoExp`

Another subclass of `HamEvo`, this class performs the Hamiltonian evolution operation using the Matrix Exponential method. For efficiency, it checks if all the Hamiltonians in the batch are diagonal and skips the computation of matrix exponentials if they are.

## Enum Definitions

### `HamEvoType`

An enumeration to represent types of Hamiltonian Evolution, including RK4, Eigenvalue Decomposition, and Exponential. It contains the corresponding classes as the enumeration values.

## Function Definitions

### `diagonalize`

A function to diagonalize a Hermitian Hamiltonian, returning eigenvalues and eigenvectors. It also performs checks to see if the Hamiltonian is already diagonal and if it's real to avoid unnecessary computations.

## Examples

The following examples show how to use the `HamiltonianEvolution` module:
<br>
<br>
Initialization of HamiltonianEvolution takes parameters (qubits, n_qubits, n_steps, hamevo_type)
<br>
Using the HamiltonianEvolution instance to evolve the state takes parameters (H, t, state)
<br>

### Example 1:
```python
import torch
import pyqtorch.modules as pyq

#Define initialization parameters
n_qubits = 2
qubits = list(range(n_qubits))

# Define the Hamiltonian H (random for this example)
H = torch.randn((2**n_qubits, 2**n_qubits), dtype=torch.cdouble)

# Make sure H is Hermitian as required for a Hamiltonian
H = (H + H.conj().T) / 2

# Define the initial state
state = pyq.uniform_state(n_qubits)

# Define the evolution time tensor
t = torch.tensor([torch.pi / 4], dtype=torch.cdouble)

# Instantiate HamiltonianEvolution with RK4 string input
hamiltonian_evolution = pyq.HamiltonianEvolution(qubits, n_qubits, 100, "RK4")

# Use the HamiltonianEvolution instance to evolve the state
output_state_rk4 = hamiltonian_evolution(H, t, state)

```



### Example 2:
```python
import torch
import pyqtorch.modules as pyq

# Define initialization parameters
n_qubits = 1
qubits = list(range(n_qubits))
n_steps = 100

# Define the Hamiltonian H for a qubit in a z-direction magnetic field
H = torch.tensor([[0.5, 0], [0, -0.5]], dtype=torch.cdouble)

# Define the initial state as |0> (you could also try with |1> or a superposition state)
state = torch.tensor([[1], [0]], dtype=torch.cdouble)

# Define the evolution time tensor
t = torch.tensor([torch.pi / 2], dtype=torch.cdouble)

# Instantiate HamiltonianEvolution with HamEvoType input
H_evol = pyq.HamiltonianEvolution(qubits, n_qubits, n_steps, pyq.HamEvoType.EIG)

# Use the HamiltonianEvolution instance to evolve the state
output_state_eig = H_evol(H, t, state)

# Print the output state
print(output_state_eig)

# Now compare the output state with the expected result: e^(-i * H * t) * |0> = [[e^(-i*t/2)], [0]]
expected_state = torch.tensor([[torch.exp(-1j * t / 2)], [0]], dtype=torch.cdouble)
print(expected_state)

# Check if the output_state is close to the expected_state
print(torch.allclose(output_state_rk4, expected_state))

```
