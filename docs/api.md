`pyqtorch` exposes `run`, `sample` and `expectation` routines with the following interface:

## run
```python
def run(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    embedding: Embedding | None = None,
) -> Tensor:
    """Sequentially apply each operation in `circuit.operations` to an input state `state`
    given current parameter values `values`, perform an optional `embedding` on `values`
    and return an output state.

    Arguments:
    circuit: A pyqtorch.QuantumCircuit instance.
    state: A torch.Tensor of shape [2, 2, ..., batch_size].
    values: A dictionary containing <'parameter_name': torch.Tensor> pairs denoting
            the current parameter values for each parameter in `circuit`.
    embedding: An optional instance of `Embedding`.
    Returns:
         A torch.Tensor of shape [2, 2, ..., batch_size]
    """
    ...
```

## sample
```python
def sample(
    circuit: QuantumCircuit,
    state: Tensor = None,
    values: dict[str, Tensor] = dict(),
    n_shots: int = 1000,
    embedding: Embedding | None = None,
) -> list[Counter]:
    """Sample from `circuit` given an input state `state` given current parameter values `values`,
       perform an optional `embedding` on `values` and return a list Counter objects mapping from
       bitstring: num_samples.

    Arguments:
    circuit: A pyqtorch.QuantumCircuit instance.
    state: A torch.Tensor of shape [2, 2, ..., batch_size].
    values: A dictionary containing <'parameter_name': torch.Tensor> pairs
            denoting the current parameter values for each parameter in `circuit`.
    n_shots: A positive int denoting the number of requested samples.
    embedding: An optional instance of `Embedding`.
    Returns:
         A list of Counter objects containing bitstring:num_samples pairs.
    """
    ...
```

## expectation

```python
def expectation(
    circuit: QuantumCircuit,
    state: Tensor,
    values: dict[str, Tensor],
    observable: Observable,
    diff_mode: DiffMode = DiffMode.AD,
    embedding: Embedding | None = None,
) -> Tensor:
    """Compute the expectation value of `circuit` given a `state`, parameter values `values`
        given an `observable` and optionally compute gradients using diff_mode.
    Arguments:
        circuit: A pyqtorch.QuantumCircuit instance.
        state: A torch.Tensor of shape [2, 2, ..., batch_size].
        values: A dictionary containing <'parameter_name': torch.Tensor> pairs
                denoting the current parameter values for each parameter in `circuit`.
        observable: A pyq.Observable instance.
        diff_mode: The differentiation mode.
        embedding: An optional instance of `Embedding`.
    Returns:
        An expectation value.
    """
    ...
```
