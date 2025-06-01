from __future__ import annotations

from functools import reduce
from operator import add

import numpy as np
import pytest
import torch
import torch.autograd.gradcheck
from torch.nn import Module

import pyqtorch as pyq
from pyqtorch.embed import ConcretizedCallable, Embedding, cos, log, sin, sqrt
from pyqtorch.primitives import Primitive
from pyqtorch.utils import ATOL_embedding


@pytest.mark.parametrize(
    "fn", ["sin", "cos", "log", "tanh", "tan", "sin", "sqrt", "square"]
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
    assert np.allclose(results[0], results[1], atol=ATOL_embedding) and np.allclose(
        results[0], results[2], ATOL_embedding
    )


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
    assert np.allclose(results[0], results[1], atol=ATOL_embedding) and np.allclose(
        results[0], results[2], atol=ATOL_embedding
    )


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
    assert np.allclose(results[0], results[1], atol=ATOL_embedding) and np.allclose(
        results[0], results[2], atol=ATOL_embedding
    )


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
        reembedded_params = embedding.reembed_tparam(all_params, new_tparam_val)
        results.append(all_params["%2"].item())
        reembedded_results.append(reembedded_params["%2"].item())
    assert all([p in ["%1", "%2"] for p in embedding.tracked_vars])
    assert "%0" not in embedding.tracked_vars
    assert np.allclose(results[0], results[1], atol=ATOL_embedding) and np.allclose(
        results[0], results[2], atol=ATOL_embedding
    )
    assert np.allclose(
        reembedded_results[0], reembedded_results[1], atol=ATOL_embedding
    ) and np.allclose(reembedded_results[0], reembedded_results[2], atol=ATOL_embedding)


def test_sample_run_expectation_grads_with_embedding() -> None:
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
    obs = pyq.Observable(pyq.Z(0))

    state = pyq.zero_state(n_qubits)

    x = torch.rand(1, requires_grad=True)
    y = torch.rand(1, requires_grad=True)

    values_ad = {"x": x, "y": y}

    wf = pyq.run(circ, state, values_ad, embedding)
    samples = pyq.sample(circ, state, values_ad, 100, embedding)
    exp_ad = pyq.expectation(
        circ, state, values_ad, obs, pyq.DiffMode.AD, embedding=embedding
    )
    assert torch.autograd.gradcheck(
        lambda x, y: pyq.expectation(
            circ, state, {"x": x, "y": y}, obs, pyq.DiffMode.AD, embedding=embedding
        ),
        (x, y),
        atol=1e-1,  # torch.autograd.gradcheck is very susceptible to small numerical errors
    )


@pytest.mark.parametrize("engine", ["torch", "jax", "numpy"])
def test_move_embedding(engine: str) -> None:
    name0, fn0 = "fn0", ConcretizedCallable("sin", ["x"], engine_name=engine)
    name1, fn1 = "fn1", ConcretizedCallable("mul", ["fn0", "y"], engine_name=engine)
    name2, fn2 = "fn2", ConcretizedCallable("mul", ["fn1", 2.0], engine_name=engine)
    name3, fn3 = "fn3", ConcretizedCallable("log", ["fn2"], engine_name=engine)
    embedding = pyq.Embedding(
        vparam_names=["x"],
        fparam_names=["y"],
        var_to_call={name0: fn0, name1: fn1, name2: fn2, name3: fn3},
        engine_name=engine,
    )
    embedding.to(device="cpu", dtype=torch.float32)
    embedding.to(device="cpu")
    embedding.to(dtype=torch.float32)


def test_reembedding_forward() -> None:

    class CustomReembed(pyq.Add):
        def __init__(self, operations: list[Module] | Primitive | pyq.Sequence):
            super().__init__(operations=operations)

        def forward(
            self,
            state: torch.Tensor,
            values: dict[str, torch.Tensor] | None = None,
            embedding: Embedding | None = None,
        ) -> torch.Tensor:
            values = values or dict()
            return reduce(
                add,
                [
                    self.run(state, values, embedding.reembed_tparam(values, val))  # type: ignore[union-attr, arg-type]
                    for val in torch.linspace(0.0, 10, 10)
                ],
            )

        def run(
            self,
            state: torch.Tensor,
            values: dict[str, torch.Tensor],
            embedding: Embedding | None = None,
        ) -> torch.Tensor:
            for op in self.operations:
                state = op(state, values)
            return state

    leaf0, native_call0 = "%0", ConcretizedCallable("sin", ["t"], {}, "torch")
    gen = pyq.Scale(pyq.Z(0), leaf0)
    custom = CustomReembed(gen)
    embed = pyq.Embedding(
        vparam_names=[],
        fparam_names=[],
        var_to_call={leaf0: native_call0},
        tparam_name="t",
    )
    wf = custom(state=pyq.zero_state(2), values={"t": torch.rand(1)}, embedding=embed)
    assert not torch.any(torch.isnan(wf))


def test_get_independent_args() -> None:
    expr: ConcretizedCallable = sqrt(sin("x")) + cos("r") * (1.0 / log("z") * "y")
    assert set(expr.independent_args) == {"x", "y", "z", "r"}


def test_sample_run_expectation_grads_with_embedding_observable(embedding_fixture):
    """
    Extended test for embedding with observables that have embedded parameters.

    This extends the existing test_sample_run_expectation_grads_with_embedding
    by testing against an observable with embedded parameters (Scale(Z(0), embedding)).

    Mathematical verification:
    ⟨O_embedded⟩ = ⟨ψ|f(x,y)·Z|ψ⟩ = f(x,y)·⟨ψ|Z|ψ⟩
    where f(x,y) = y * sin(x)
    """
    import torch
    import torch.autograd.gradcheck

    import pyqtorch as pyq
    from pyqtorch.composite import Scale

    embedding = embedding_fixture

    # Create circuit with embedded parameters
    rx = pyq.RX(0, param_name="sin_x")  # θ = sin(x)
    cry = pyq.CRY(0, 1, param_name="mul_sinx_y")  # θ = y*sin(x)
    phase = pyq.PHASE(1, param_name="mul_sinx_y")  # θ = y*sin(x)
    ry = pyq.RY(1, param_name="mul_sinx_y")  # θ = y*sin(x)
    cnot = pyq.CNOT(1, 2)

    ops = [rx, cry, phase, ry, cnot]
    n_qubits = 3
    circ = pyq.QuantumCircuit(n_qubits, ops)

    # Test observables
    obs_simple = pyq.Observable([pyq.Z(0)])  # Simple observable
    obs_embedded = pyq.Observable(
        [Scale(pyq.Z(0), param_name="mul_sinx_y")]
    )  # Embedded observable

    state = pyq.zero_state(n_qubits)

    # Parameter values
    x = torch.rand(1, requires_grad=True)
    y = torch.rand(1, requires_grad=True)
    values_ad = {"x": x, "y": y}

    # Test 1: Simple observable expectation
    exp_simple = pyq.expectation(
        circ, state, values_ad, obs_simple, pyq.DiffMode.AD, embedding=embedding
    )

    # Test 2: Embedded observable expectation
    exp_embedded = pyq.expectation(
        circ, state, values_ad, obs_embedded, pyq.DiffMode.AD, embedding=embedding
    )
    # y * sin(x)
    # Test 3: Manual verification
    embedded_values = embedding(values_ad)
    scaling_factor = embedded_values["mul_sinx_y"]

    # Manual calculation: f(x,y) * ⟨ψ|Z|ψ⟩
    exp_manual = scaling_factor * pyq.expectation(
        circ, state, embedded_values, obs_simple, pyq.DiffMode.AD
    )

    assert torch.allclose(
        exp_embedded, exp_manual, atol=1e-6
    ), "Embedded observable expectation should equal scaling_factor * simple_expectation"

    # Test 4: Gradient verification for embedded observable
    def loss_embedded(x_val, y_val):
        vals = {"x": x_val, "y": y_val}
        return pyq.expectation(
            circ, state, vals, obs_embedded, pyq.DiffMode.AD, embedding=embedding
        )

    # Verify gradients are computed correctly
    assert torch.autograd.gradcheck(
        loss_embedded, (x, y), atol=1e-1
    ), "Gradient check failed for embedded observable"

    # Test 5: Consistency check - observable evaluation equivalence
    # Method A: Observable with embedding
    exp_obs_embed = obs_embedded.expectation(state, values_ad, embedding)

    # Method B: Observable with pre-evaluated parameters
    exp_obs_eval = obs_embedded.expectation(state, embedded_values)

    assert torch.allclose(
        exp_obs_embed, exp_obs_eval, atol=1e-6
    ), "Observable expectation with embedding should match pre-evaluated expectation"

    # Test 6: Verify the mathematical relationship: ⟨f(x,y)·Z⟩ = f(x,y)·⟨Z⟩
    # Get the final state after circuit execution
    final_state = circ(state, values_ad, embedding)

    # Simple expectation ⟨Z⟩
    simple_exp_value = obs_simple.expectation(final_state)

    # Embedded expectation should equal scaling_factor * simple_expectation
    expected_embedded_value = scaling_factor * simple_exp_value
    actual_embedded_value = obs_embedded.expectation(final_state, embedded_values)

    assert torch.allclose(
        expected_embedded_value, actual_embedded_value, atol=1e-6
    ), "Mathematical relationship ⟨f(x,y)·Z⟩ = f(x,y)·⟨Z⟩ violated"

    print("✓ All embedding observable tests passed")


def test_embedding_full_support_consistency(embedding_fixture):
    """
    Test embedding consistency with full_support parameter in tensor methods.

    SIMPLIFIED VERSION: Uses only pyqtorch's built-in methods to avoid
    manual tensor manipulation issues.

    Mathematical verification:
    U_expanded(f(x,y)) = expand_operator(U_local(f(x,y)), local_support, full_support)
    """
    import torch

    import pyqtorch as pyq
    from pyqtorch.apply import apply_operator
    from pyqtorch.utils import expand_operator, random_state

    embedding = embedding_fixture

    # Test parameters
    batch_size = 2
    x = torch.rand(batch_size)
    y = torch.rand(batch_size)
    values = {"x": x, "y": y}

    # Create a gate on subset of qubits
    gate = pyq.CRX(0, 1, param_name="mul_sinx_y")  # Acts on qubits 0,1
    local_support = gate.qubit_support
    full_support = (0, 1, 2, 3)  # Expand to 4 qubits

    # Test 1: tensor() method with full_support vs manual expansion
    # Method A: Direct embedding with full_support
    tensor_full_embed = gate.tensor(values, embedding, full_support=full_support)

    # Method B: Local tensor + manual expansion
    tensor_local_embed = gate.tensor(values, embedding)
    tensor_full_manual = expand_operator(
        tensor_local_embed, local_support, full_support
    )

    assert torch.allclose(
        tensor_full_embed, tensor_full_manual, atol=1e-6
    ), "Full support tensor with embedding should match manual expansion"

    # Test 2: Consistency with pre-evaluation
    embedded_values = embedding(values)

    # Method C: Pre-evaluation with full_support
    tensor_full_eval = gate.tensor(embedded_values, full_support=full_support)

    assert torch.allclose(
        tensor_full_embed, tensor_full_eval, atol=1e-6
    ), "Full support tensor embedding should match pre-evaluated full support tensor"

    # Test 3: SIMPLIFIED state application using pyqtorch's apply_operator
    # Create state for full support
    state = random_state(len(full_support), batch_size)

    # Method A: Apply tensor with embedding using apply_operator
    result_embed = apply_operator(state.clone(), tensor_full_embed, full_support)

    # Method B: Apply manually expanded tensor using apply_operator
    result_manual = apply_operator(state.clone(), tensor_full_manual, full_support)

    assert torch.allclose(
        result_embed, result_manual, atol=1e-6
    ), "apply_operator results should be identical for embedding and manual expansion"

    # Test 4: End-to-end verification with circuit application
    # Create a circuit that uses the gate
    circuit = pyq.QuantumCircuit(len(full_support), [gate])

    # Method A: Circuit with embedding
    result_circuit_embed = circuit(state.clone(), values, embedding)

    # Method B: Circuit with pre-evaluated values
    result_circuit_eval = circuit(state.clone(), embedded_values)

    assert torch.allclose(
        result_circuit_embed, result_circuit_eval, atol=1e-6
    ), "Circuit execution with embedding should match pre-evaluated circuit"

    # Test 5: Verify expand_operator consistency with different supports
    # Test with minimal support vs full support
    minimal_gate = pyq.CRX(0, 1, param_name="mul_sinx_y")

    # Tensor on minimal support
    tensor_minimal = minimal_gate.tensor(values, embedding)

    # Expand to full support manually
    tensor_expanded = expand_operator(tensor_minimal, (0, 1), full_support)

    # Direct tensor on full support
    tensor_direct_full = minimal_gate.tensor(
        values, embedding, full_support=full_support
    )

    assert torch.allclose(
        tensor_expanded, tensor_direct_full, atol=1e-6
    ), "Manual expansion should match direct full_support tensor generation"

    print("✓ Full support embedding consistency tests passed")


def test_embedding_diagonal_operations(embedding_fixture):
    """
    Test embedding functionality with diagonal operations.

    Verifies that embeddings work correctly with diagonal gates and that
    the diagonal=True parameter produces consistent results.

    Mathematical verification for diagonal gates:
    diag(U(f(x,y))) should be consistent with dense(U(f(x,y)))
    """
    import torch

    import pyqtorch as pyq
    from pyqtorch.composite import Scale
    from pyqtorch.utils import random_state, todense_tensor

    embedding = embedding_fixture

    # Test parameters
    n_qubits = 3
    batch_size = 2
    x = torch.rand(batch_size)
    y = torch.rand(batch_size)
    values = {"x": x, "y": y}

    # Test diagonal gates (Z, PHASE, S, T, etc.)
    diagonal_gates = [
        pyq.RZ(0, param_name="mul_sinx_y"),
        pyq.PHASE(1, param_name="mul_sinx_y"),
        Scale(pyq.Z(2), "mul_sinx_y"),
    ]

    for gate in diagonal_gates:
        # Test 1: Verify gate is recognized as diagonal
        assert gate.is_diagonal, f"Gate {gate} should be diagonal"

        # Test 2: Dense tensor method
        tensor_dense_embed = gate.tensor(values, embedding, diagonal=False)
        tensor_dense_eval = gate.tensor(embedding(values), diagonal=False)

        assert torch.allclose(
            tensor_dense_embed, tensor_dense_eval, atol=1e-6
        ), f"Dense tensor embedding consistency failed for {gate}"

        # Test 3: Diagonal tensor method
        tensor_diag_embed = gate.tensor(values, embedding, diagonal=True)
        tensor_diag_eval = gate.tensor(embedding(values), diagonal=True)

        assert torch.allclose(
            tensor_diag_embed, tensor_diag_eval, atol=1e-6
        ), f"Diagonal tensor embedding consistency failed for {gate}"

        # Test 4: Diagonal vs Dense consistency
        dense_from_diag = todense_tensor(tensor_diag_embed)

        assert torch.allclose(
            tensor_dense_embed, dense_from_diag, atol=1e-6
        ), f"Diagonal and dense tensors should be equivalent for {gate}"

        # Test 5: Application to states
        state = random_state(n_qubits, batch_size)

        # Using dense tensor
        result_dense = gate(state.clone(), values, embedding)

        # Manual application with diagonal tensor (for verification)
        if hasattr(gate, "qubit_support"):
            from pyqtorch.apply import apply_operator

            result_manual = apply_operator(
                state.clone(), tensor_dense_embed, gate.qubit_support
            )

            assert torch.allclose(
                result_dense, result_manual, atol=1e-6
            ), f"Dense application should match manual tensor application for {gate}"

    print("✓ Diagonal operations embedding tests passed")
