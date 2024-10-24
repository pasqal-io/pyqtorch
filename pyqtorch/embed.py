from __future__ import annotations

from importlib import import_module
from logging import getLogger
from typing import Any, Tuple

from numpy.typing import ArrayLike, DTypeLike
from torch import Tensor

logger = getLogger(__name__)


ARRAYLIKE_FN_MAP = {
    "torch": ("torch", "tensor"),
    "jax": ("jax.numpy", "array"),
    "numpy": ("numpy", "array"),
}


DEFAULT_JAX_MAPPING = {
    "mul": ("jax.numpy", "multiply"),
    "sub": ("jax.numpy", "subtract"),
    "div": ("jax.numpy", "divide"),
}
DEFAULT_TORCH_MAPPING = {"hs": ("pyqtorch.utils", "heaviside")}
DEFAULT_NUMPY_MAPPING = {
    "mul": ("numpy", "multiply"),
    "sub": ("numpy", "subtract"),
    "div": ("numpy", "divide"),
}

DEFAULT_INSTRUCTION_MAPPING = {
    "torch": DEFAULT_TORCH_MAPPING,
    "jax": DEFAULT_JAX_MAPPING,
    "numpy": DEFAULT_NUMPY_MAPPING,
}


class ConcretizedCallable:
    """Transform an abstract function name and arguments into
        a callable in a linear algebra engine which can be evaluated
        using user input.
    Arguments:
        call_name: The name of the function.
        abstract_args: A list of arguments to the function,
                       can be numeric types for constants or strings for parameters
        instruction_mapping: A dict containing user-passed mappings from a function name
                            to its implementation.
        engine_name: The name of the framework to use.
        device: Which device to use.

    Example:
    ```
    import torch

    from pyqtorch.embed import ConcretizedCallable


    In [11]: call = ConcretizedCallable('sin', ['x'], engine_name='numpy')
    In [12]: call({'x': 0.5})
    Out[12]: 0.479425538604203

    In [13]: call = ConcretizedCallable('sin', ['x'], engine_name='torch')
    In [14]: call({'x': torch.rand(1)})
    Out[14]: tensor([0.5531])

    In [15]: call = ConcretizedCallable('sin', ['x'], engine_name='jax')
    In [16]: call({'x': 0.5})
    Out[16]: Array(0.47942555, dtype=float32, weak_type=True)
    ```



    """

    def __init__(
        self,
        call_name: str = "",
        abstract_args: list[str | float | int | complex | ConcretizedCallable] = ["x"],
        instruction_mapping: dict[str, Tuple[str, str]] | None = None,
        engine_name: str = "torch",
        device: str = "cpu",
        dtype: Any = None,
    ) -> None:
        instruction_mapping = instruction_mapping or dict()
        instruction_mapping = {
            **instruction_mapping,
            **DEFAULT_INSTRUCTION_MAPPING[engine_name],
        }
        self.call_name = call_name
        self.abstract_args = abstract_args
        self.engine_name = engine_name
        self._device = device
        self._dtype = dtype
        self.engine_call = None
        engine = None
        if not all(
            [
                isinstance(arg, (str, float, int, complex, Tensor, ConcretizedCallable))
                for arg in abstract_args
            ]
        ):
            raise TypeError(
                "Only str, float, int, complex, Tensor or ConcretizedCallable type elements \
                are supported for abstract_args"
            )
        try:
            engine_name, fn_name = ARRAYLIKE_FN_MAP[engine_name]
            engine = import_module(engine_name)
            self.arraylike_fn = getattr(engine, fn_name)
        except (ModuleNotFoundError, ImportError) as e:
            logger.error(f"Unable to import {engine_name} due to {e}.")

        try:
            self.engine_call = getattr(engine, call_name, None)
            if self.engine_call is None:
                mod, fn = instruction_mapping[call_name]
                self.engine_call = getattr(import_module(mod), fn)
        except (ImportError, KeyError) as e:
            logger.error(
                f"Requested function {call_name} can not be imported from {engine_name} and is"
                + f" not in instruction_mapping {instruction_mapping} due to {e}."
            )

    def evaluate(self, inputs: dict[str, ArrayLike] | None = None) -> ArrayLike:
        arraylike_args = []
        inputs = inputs or dict()
        for symbol_or_numeric in self.abstract_args:
            if isinstance(symbol_or_numeric, ConcretizedCallable):
                arraylike_args.append(symbol_or_numeric(inputs))
            if isinstance(symbol_or_numeric, (float, int, Tensor)):
                arraylike_args.append(
                    self.arraylike_fn(symbol_or_numeric, device=self.device)
                )
            elif isinstance(symbol_or_numeric, str):
                arraylike_args.append(inputs[symbol_or_numeric])
        return self.engine_call(*arraylike_args)  # type: ignore[misc]

    @classmethod
    def _get_independent_args(cls, cc: ConcretizedCallable) -> set:
        out: set = set()
        if len(cc.abstract_args) == 1 and isinstance(cc.abstract_args[0], str):
            return set([cc.abstract_args[0]])
        else:
            for arg in cc.abstract_args:
                if isinstance(arg, ConcretizedCallable):
                    res = cls._get_independent_args(arg)
                    out = out.union(res)
                else:
                    if isinstance(arg, str):
                        out.add(arg)
        return out

    @property
    def independent_args(self) -> list:
        return list(self._get_independent_args(self))

    def __call__(self, inputs: dict[str, ArrayLike] | None = None) -> ArrayLike:
        return self.evaluate(inputs)

    def __mul__(
        self, other: str | int | float | complex | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("mul", [self, other])

    def __rmul__(
        self, other: str | int | float | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("mul", [other, self])

    def __add__(
        self, other: str | int | float | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("add", [self, other])

    def __radd__(
        self, other: str | int | float | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("add", [other, self])

    def __sub__(
        self, other: str | int | float | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("sub", [self, other])

    def __rsub__(
        self, other: str | int | float | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("sub", [other, self])

    def __pow__(
        self, other: str | int | float | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("pow", [self, other])

    def __rpow__(
        self, other: str | int | float | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("pow", [other, self])

    def __truediv__(
        self, other: str | int | float | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("div", [self, other])

    def __rtruediv__(
        self, other: str | int | float | ConcretizedCallable
    ) -> ConcretizedCallable:
        return ConcretizedCallable("div", [other, self])

    def __repr__(self) -> str:
        return f"{self.call_name}({self.abstract_args})"

    def __neg__(self) -> ConcretizedCallable:
        return -1 * self

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> Any:
        return self._dtype

    def to(self, *args: Any, **kwargs: Any) -> ConcretizedCallable:
        self._device = kwargs.get("device", None)
        self._dtype = kwargs.get("dtype", None)
        return self


def init_param(
    engine_name: str, trainable: bool = True, device: str = "cpu"
) -> ArrayLike:
    engine = import_module(engine_name)
    if engine_name == "jax":
        return engine.random.uniform(engine.random.PRNGKey(42), shape=(1,))
    elif engine_name == "torch":
        return engine.rand(1, requires_grad=trainable, device=device)
    elif engine_name == "numpy":
        return engine.random.uniform(0, 1)


def sin(x: str | ConcretizedCallable):
    return ConcretizedCallable("sin", [x])


def cos(x: str | ConcretizedCallable):
    return ConcretizedCallable("cos", [x])


def log(x: str | ConcretizedCallable):
    return ConcretizedCallable("log", [x])


def tan(x: str | ConcretizedCallable):
    return ConcretizedCallable("tan", [x])


def tanh(x: str | ConcretizedCallable):
    return ConcretizedCallable("tanh", [x])


def sqrt(x: str | ConcretizedCallable):
    return ConcretizedCallable("sqrt", [x])


class Embedding:
    """A class relating variational and feature parameters used in ConcretizedCallable instances to
    parameter names used in gates.

    Arguments:
        vparam_names: A list of variational parameters.
        fparam_names: A list of feature parameters.
        var_to_call: A dict mapping from <`parameter_name`: ConcretizedCallable> pairs,.
        tparam_name: Optional name for a time parameter.
        engine_name: The name of the linear algebra engine.
        device: The device to use

    Example:
    ```
    from __future__ import annotations

    import numpy as np
    import pytest
    import torch
    import torch.autograd.gradcheck

    import pyqtorch as pyq
    from pyqtorch.embed import ConcretizedCallable, Embedding
    name0, fn0 = "fn0", ConcretizedCallable("sin", ["x"])
    name1, fn1 = "fn1", ConcretizedCallable("mul", ["fn0", "y"])
    name2, fn2 = "fn2", ConcretizedCallable("mul", ["fn1", 2.0])
    name3, fn3 = "fn3", ConcretizedCallable("log", ["fn2"])
    embedding = pyq.Embedding(
        vparam_names=["x"],
        fparam_names=["y"],
        var_to_call={name0: fn0, name1: fn1, name2: fn2, name3: fn3},
    )
    rx = pyq.RX(0, param_name=name0)
    cry = pyq.CRY(0, 1, param_name=name1)
    phase = pyq.PHASE(1, param_name=name2)
    ry = pyq.RY(1, param_name=name3)
    cnot = pyq.CNOT(1, 2)
    ops = [rx, cry, phase, ry, cnot]
    n_qubits = 3
    circ = pyq.QuantumCircuit(n_qubits, ops)
    obs = pyq.Observable([pyq.Z(0)])

    state = pyq.zero_state(n_qubits)

    x = torch.rand(1, requires_grad=True)
    y = torch.rand(1, requires_grad=True)

    values_ad = {"x": x, "y": y}
    embedded_params = embedding(values_ad)
    wf = pyq.run(circ, state, embedded_params, embedding)
    ```
    """

    def __init__(
        self,
        vparam_names: list[str] = [],
        fparam_names: list[str] = [],
        var_to_call: dict[str, ConcretizedCallable] | None = None,
        tparam_name: str | None = None,
        engine_name: str = "torch",
        device: str = "cpu",
    ) -> None:
        var_to_call = var_to_call or dict()
        self.vparams = {
            vp: init_param(engine_name, trainable=True, device=device)
            for vp in vparam_names
        }
        self.fparam_names: list[str] = fparam_names
        self.tparam_name = tparam_name
        self.var_to_call: dict[str, ConcretizedCallable] = var_to_call
        self._dtype: DTypeLike = None
        self.tracked_vars: list[str] = []
        self._device = device
        self._tracked_vars_identified = False
        self.engine_name = engine_name

    @property
    def root_param_names(self) -> list[str]:
        return list(self.vparams.keys()) + self.fparam_names

    def embed_all(
        self,
        inputs: dict[str, ArrayLike] | None = None,
    ) -> dict[str, ArrayLike]:
        """The standard embedding of all intermediate and leaf parameters.
        Include the root_params, i.e., the vparams and fparams original values
        to be reused in computations.
        """
        inputs = inputs or dict()
        for intermediate_or_leaf_var, engine_callable in self.var_to_call.items():
            # We mutate the original inputs dict and include intermediates and leaves.
            if not self._tracked_vars_identified:
                # we do this only on the first embedding call
                if self.tparam_name and any(
                    [
                        p in [self.tparam_name] + self.tracked_vars
                        for p in engine_callable.abstract_args
                    ]  # we check if any parameter in the callables args is time
                    # or depends on an intermediate variable which itself depends on time
                ):
                    self.tracked_vars.append(intermediate_or_leaf_var)
                    # we remember which parameters depend on time
            inputs[intermediate_or_leaf_var] = engine_callable(inputs)
        self._tracked_vars_identified = True
        return inputs

    def reembed_tparam(
        self,
        embedded_params: dict[str, ArrayLike],
        tparam_value: ArrayLike,
    ) -> dict[str, ArrayLike]:
        """Receive already embedded params containing intermediate and leaf parameters
        and recalculate the those which are dependent on `tparam_name` using the new value
        `tparam_value`.
        """
        if self.tparam_name is None:
            raise ValueError(
                "`reembed_param` requires a `tparam_name` to be passed\
                              when initializing the `Embedding` class"
            )
        embedded_params[self.tparam_name] = tparam_value
        for time_dependent_param in self.tracked_vars:
            embedded_params[time_dependent_param] = self.var_to_call[
                time_dependent_param
            ](embedded_params)
        return embedded_params

    def __call__(
        self, inputs: dict[str, ArrayLike] | None = None
    ) -> dict[str, ArrayLike]:
        """Functional version of legacy embedding: Return a new dictionary\
        with all embedded parameters."""
        return self.embed_all(inputs or dict())

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

    def to(self, *args: Any, **kwargs: Any) -> Embedding:
        if self.engine_name == "torch":
            # we only support this for torch for now
            self.vparams = {p: t.to(*args, **kwargs) for p, t in self.vparams.items()}
            self.var_to_call = {
                p: call.to(*args, **kwargs) for p, call in self.var_to_call.items()
            }
            # Dtype and device have to be passes as kwargs
            self._dtype = kwargs.get("dtype", self._dtype)
            self._device = kwargs.get("device", self._device)
        return self
