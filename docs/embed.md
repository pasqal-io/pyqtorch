By default, a `Parametric` operation expects a `values` dict with a
value for its parameter in the forward-pass when initialized using a `str` parameter.
## Using arbitrary expressions as parameters
`pyqtorch` allows for using arbitary expressions as parameters, for instance `sin(x)` where `x` can be
a FeatureParameter. To do so, a name has to be assigned to the outcome of the evaluation of `sin(x)` and
supplied to the `pyq.QuantumCircuit` within an instance of `Embedding`.

### Creating parameter expressions using `ConcretizedCallable`
`pyq.ConcretizedCallable` expects a name for a function and a list of arguments
```python exec="on" source="material-block" html="1" session="expr"
import torch
import pyqtorch as pyq
sin_x, sin_x_fn = 'sin_x', pyq.ConcretizedCallable(call_name = 'sin', abstract_args=['x'])
# We can now evaluate sin_x_fn using a values dict
x = torch.rand(1, requires_grad=True)
values = {'x': x}
result = sin_x_fn(values)
print(torch.autograd.grad(result, x, torch.ones_like(result))[0])
```

### Interfacing `ConcretizedCallable` with QuantumCircuit parameters via the `Embedding` class
Lets use `sin_x` in another callable, so our gate will be parametrized by the result of the expression `y * sin(x)` where `y` is trainable and `x` is a feature parameter.
We can tell `pyq` how to associate each callable with its underlying parameters via the `Embedding` class which expects arguments regarding what are trainable and non-trainable symbols.

```python exec="on" source="material-block" html="1" session="expr"

mul_sinx_y, mul_sinx_y_fn = 'mul_sinx_y', pyq.ConcretizedCallable(call_name = 'mul', abstract_args=['sin_x', 'y'])
embedding = pyq.Embedding(vparam_names=['y'], fparam_names=['x'], var_to_call={sin_x: sin_x_fn, mul_sinx_y: mul_sinx_y_fn})
circ = pyq.QuantumCircuit(1, [pyq.RX(0, mul_sinx_y)])
state= pyq.zero_state(1)
y = torch.rand(1, requires_grad=True)
values = {'x': x, 'y': y}
obs = pyq.Observable([pyq.Z(0)])
expval = pyq.expectation(circuit=circ, state=state, values=values, observable=obs, diff_mode=pyq.DiffMode.AD, embedding=embedding)
print(torch.autograd.grad(expval, (x, y), torch.ones_like(expval)))
```

### Tracking and Reembedding a tracked parameter
For specific usecases, a `tparam` argument can be passed to the `Embedding` which tells the class to track the
computations depending on it which enables for their efficient recomputation given different
values for `tparam`.

```python exec="on" source="material-block" html="1" session="expr"
v_params = ["theta"]
f_params = ["x"]
tparam = "t"
leaf0, native_call0 = "%0", pyq.ConcretizedCallable(
    "mul", ["x", "theta"], {}
)
leaf1, native_call1 = "%1", pyq.ConcretizedCallable(
    "mul", ["t", "%0"], {}
)

leaf2, native_call2 = "%2", pyq.ConcretizedCallable("sin", ["%1"], {})
embedding = pyq.Embedding(
    v_params,
    f_params,
    var_to_call={leaf0: native_call0, leaf1: native_call1, leaf2: native_call2},
    tparam_name=tparam,
)
inputs = {
    "x": torch.rand(1),
    "theta": torch.rand(1),
    tparam: torch.rand(1),
}
all_params = embedding.embed_all(inputs)
print(f'{leaf2} value before reembedding: {all_params[leaf2]}')
new_tparam_val = torch.rand(1)
reembedded_params = embedding.reembed_tparam(all_params, new_tparam_val)
print(f'{leaf2} value after reembedding: {reembedded_params[leaf2]}')
```
### See the docstrings for more details and examples:
#### ConcretizedCallable
::: pyqtorch.embed.ConcretizedCallable
#### Embedding
::: pyqtorch.embed.Embedding
