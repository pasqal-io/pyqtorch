from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.autograd.gradcheck

import pyqtorch as pyq
from pyqtorch.embed import ConcretizedCallable, Embedding


@pytest.mark.parametrize(
    "fn", ["sin", "cos", "log", "tanh", "tan", "acos", "sin", "sqrt", "square"]
)
def test_univariate(fn: str) -> None:
    results = []
    x = np.random.uniform(0, 1)
    for engine_name in ["jax", "torch", "numpy"]:
        native_call = ConcretizedCallable(fn, ["x"], {}, engine_name)
        native_result = native_call(
            {"x": (torch.tensor(x) if engine_name == "torch" else x)}
        )
        results.append(native_result.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])


@pytest.mark.parametrize("fn", ["mul", "add", "div", "sub"])
def test_multivariate(fn: str) -> None:
    results = []
    x = np.random.randn(1)
    y = np.random.randn(1)
    for engine_name in ["jax", "torch", "numpy"]:
        native_call = ConcretizedCallable(fn, ["x", "y"], {}, engine_name)
        native_result = native_call(
            {
                "x": torch.tensor(x) if engine_name == "torch" else x,
                "y": torch.tensor(y) if engine_name == "torch" else y,
            }
        )
        results.append(native_result.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])


def test_embedding() -> None:
    x = np.random.uniform(0, 1)
    theta = np.random.uniform(0, 1)
    results = []
    for engine_name in ["jax", "torch", "numpy"]:
        v_params = ["theta"]
        f_params = ["x"]
        leaf0, native_call0 = "%0", ConcretizedCallable(
            "mul", ["x", "theta"], {}, engine_name
        )
        embedding = Embedding(
            v_params,
            f_params,
            var_to_call={leaf0: native_call0},
            engine_name=engine_name,
        )
        inputs = {
            "x": (torch.tensor(x) if engine_name == "torch" else x),
            "theta": (torch.tensor(theta) if engine_name == "torch" else theta),
        }
        eval_0 = embedding.var_to_call["%0"](inputs)
        results.append(eval_0.item())
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])


def test_reembedding() -> None:
    x = np.random.uniform(0, 1)
    theta = np.random.uniform(0, 1)
    t = np.random.uniform(0, 1)
    t_reembed = np.random.uniform(0, 1)
    results = []
    reembedded_results = []
    for engine_name in ["jax", "torch", "numpy"]:
        v_params = ["theta"]
        f_params = ["x"]
        tparam = "t"
        leaf0, native_call0 = "%0", ConcretizedCallable(
            "mul", ["x", "theta"], {}, engine_name
        )
        leaf1, native_call1 = "%1", ConcretizedCallable(
            "mul", ["t", "%0"], {}, engine_name
        )

        leaf2, native_call2 = "%2", ConcretizedCallable("sin", ["%1"], {}, engine_name)
        embedding = Embedding(
            v_params,
            f_params,
            var_to_call={leaf0: native_call0, leaf1: native_call1, leaf2: native_call2},
            tparam_name=tparam,
            engine_name=engine_name,
        )
        inputs = {
            "x": (torch.tensor(x) if engine_name == "torch" else x),
            "theta": (torch.tensor(theta) if engine_name == "torch" else theta),
            "t": (torch.tensor(t) if engine_name == "torch" else t),
        }
        all_params = embedding.embed_all(inputs)
        new_tparam_val = (
            torch.tensor(t_reembed) if engine_name == "torch" else t_reembed
        )
        reembedded_params = embedding.reembed_time(all_params, new_tparam_val)
        results.append(all_params["%2"].item())
        reembedded_results.append(reembedded_params["%2"].item())
    assert all([p in ["%1", "%2"] for p in embedding.time_dependent_vars])
    assert "%0" not in embedding.time_dependent_vars
    assert np.allclose(results[0], results[1]) and np.allclose(results[0], results[2])
    assert np.allclose(reembedded_results[0], reembedded_results[1]) and np.allclose(
        reembedded_results[0], reembedded_results[2]
    )


@pytest.mark.parametrize("diff_mode", [pyq.DiffMode.AD])
def test_sample_run_expectation_grads_with_embedding(diff_mode) -> None:
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
    obs = pyq.Observable(n_qubits, [pyq.Z(0)])

    state = pyq.zero_state(n_qubits)

    x = torch.rand(1, requires_grad=True)
    y = torch.rand(1, requires_grad=True)

    values_ad = {"x": x, "y": y}
    embedded_params = embedding(values_ad)
    wf = pyq.run(circ, state, embedded_params, embedding)
    samples = pyq.sample(circ, state, embedded_params, 100, embedding)
    exp_ad = pyq.expectation(circ, state, embedded_params, obs, diff_mode, embedding)
    assert torch.autograd.gradcheck(
        lambda x, y: pyq.expectation(
            circ, state, {"x": x, "y": y}, obs, diff_mode, embedding
        ),
        (x, y),
        atol=0.2,
    )
