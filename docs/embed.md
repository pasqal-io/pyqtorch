By default, a `Parametric` operation expects a `values` dict with a
value for its parameter in the forward-pass when initialized using a `str` parameter.
## Using arbitrary expressions as parameters
`pyqtorch` allows for using arbitary expressions as parameters, for instance `sin(x)` where `x` can be
a FeatureParameter. To do so, a name has to be assigned to the outcome of the evaluation of `sin(x)` and
supplied to the `pyq.QuantumCircuit` within an instance of `Embedding`.

### Using pyq.torch_call to create torch callables
`pyq.torch_call` expects a name for a function and a list of arguments
```python exec="on" source="material-block" html="1" session="expr"
import torch
import pyqtorch as pyq
sin_x, sin_x_fn = 'sin_x', pyq.torch_call(abstract_fn = 'sin', args=['x'])
# We can now evaluate sin_x_fn using a values dict
x = torch.rand(1, requires_grad=True)
values = {'x': x}
result = sin_x_fn(values)
print(torch.autograd.grad(result, x, torch.ones_like(result))[0])
```

### Wrapping torch_calls in the Embedding class
Now, we can pass `sin_x : sin_x_fn` in form of a dict to the `Embedding`
along with information what are trainable and non-trainable symbols.

```python exec="on" source="material-block" html="1" session="expr"

embedding = pyq.Embedding([], ['x'], {sin_x: sin_x_fn})
circ = pyq.QuantumCircuit(1, [pyq.RX(0, sin_x)])
state= pyq.zero_state(1)
expval = pyq.expectation(circuit=circ, state=state, values=values, observable= pyq.Observable(1, [pyq.Z(0)]),diff_mode=pyq.DiffMode.AD,embedding=embedding)

print(torch.autograd.grad(expval, x, torch.ones_like(expval))[0])
```
