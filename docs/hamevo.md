# Hamiltonian Evolution Module Documentation

## Overview

The Hamiltonian Evolution (`HamiltonianEvolution`) module is designed for performing quantum operations using different Hamiltonian evolution strategies, such as 4th order Runge-Kutta (RK4), Eigenvalue Decomposition, and Matrix Exponential.

This module also features a function `diagonalize()` that performs an eigenvalue decomposition on a given Hamiltonian. This function checks if the input Hamiltonian is already diagonal and real before computing the decomposition, thus saving computational resources when the checks are met.

## Class Definitions

### `HamiltonianEvolution`

This class is a PyTorch module designed to encapsulate Hamiltonian Evolution operations. The evolution operation performed by the module depends on the strategy specified through the `hamevo_type` attribute, which can be a member of the `HamEvoType` enumeration or an equivalent string ("RK4", "EIG", "EXP").

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

The following example shows how to use the `HamiltonianEvolution` module:

```python
# Instantiate HamiltonianEvolution with RK4 string input
hamiltonian_evolution = HamiltonianEvolution(qubits, n_qubits, 100, "RK4")
# Use the HamiltonianEvolution instance to evolve the state
output_state = hamiltonian_evolution(H, t, state)

# Instantiate HamiltonianEvolution with HamEvoType input
H_evol = HamiltonianEvolution(qubits, n_qubits, 100, HamEvoType.EIG)
# Use the HamiltonianEvolution instance to evolve the state
output_state = H_evol(H, t, state)
