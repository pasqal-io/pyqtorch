from __future__ import annotations

import torch
from torch import Tensor

from pyqtorch.time_dependent.integrators.krylov import KrylovIntegrator


class Krylov(KrylovIntegrator):
    """Krylov subspace method for solving Schrodinger equation."""

    def integrate(self, t0: float, t1: float, y: Tensor) -> Tensor:
        dt = t1 - t0
        y = self.step(dt, t1, y)
        return y

    def step(
        self,
        dt: float,
        t: float,
        state: Tensor,
    ) -> Tensor:

        # ensure Krylov solver is passed only 2D tensors
        if state.ndim != 2:
            raise ValueError(
                f"Krylov solver expects a 2D state tensor with a batch dimension"
                f" at index 1 (expected state.ndim == 2; got state.ndim == {state.ndim})."
            )

        # move trailing batch dimension to leading batch dimension
        state = state.T.unsqueeze(-1)

        def exponentiate() -> tuple[torch.Tensor, bool]:
            # approximate next iteration by modifying T, and unmodifying
            T[:, i - 1, i] = 0
            T[:, i + 1, i] = 1
            exp = torch.linalg.matrix_exp(-1j * dt * T[:, : i + 2, : i + 2].clone())
            T[:, i - 1, i] = T[:, i, i - 1]
            T[:, i + 1, i] = 0

            # compute errors
            e1 = abs(exp[:, i, 0])
            e2 = abs(exp[:, i + 1, 0]) * n

            # compute batch convergence error
            error = (e1 * e2) / (e1 - e2)
            use_e2_idx = e1 > 10 * e2
            use_e1_idx = e2 > e1
            error[use_e2_idx] = e2[use_e2_idx]
            error[use_e1_idx] = e1[use_e1_idx]

            # set convergence criteria based on maximum error in batch
            converged = error.max() < self.options.exp_tolerance
            return exp[:, :, 0], converged

        lanczos_vectors = [state]
        T = torch.zeros(
            (state.shape[0], self.options.max_krylov + 1, self.options.max_krylov + 1),
            dtype=state.dtype,
        )

        # step 0 of the loop
        v = torch.matmul(self.H(t), state)
        a = torch.matmul(v.conj().mT, state)
        a_scalar = a.squeeze((1, 2))

        n = torch.linalg.vector_norm(v.squeeze(2), dim=1)
        T[:, 0, 0] = a_scalar
        v = v - a * state

        for i in range(1, self.options.max_krylov):

            # this block should not be executed in step 0
            b = torch.linalg.vector_norm(v.squeeze(2), dim=1).view(-1, 1, 1)
            b_scalar = b.squeeze((1, 2))

            if b_scalar.max() < self.options.norm_tolerance:
                exp = torch.linalg.matrix_exp(-1j * dt * T[:, :i, :i])
                weights = exp[:, :, 0]
                converged = True
                break

            T[:, i, i - 1] = b_scalar
            T[:, i - 1, i] = b_scalar
            state = v / b

            lanczos_vectors.append(state)
            weights, converged = exponentiate()

            if converged:
                break

            v = torch.matmul(self.H(t), state)
            a = torch.matmul(v.conj().mT, state)

            n = torch.linalg.vector_norm(v.squeeze(2), dim=1)
            T[:, i, i] = a_scalar
            v = v - a * lanczos_vectors[i] - b * lanczos_vectors[i - 1]

        if not converged:
            raise RecursionError(
                "Exponentiation algorithm did not converge \
                to precision in allotted number of steps."
            )

        lanczos_vector_stack = torch.stack(lanczos_vectors, dim=1)
        n_lanczos = lanczos_vector_stack.shape[1]

        # take weighted combination of lanczos vectors and
        # move the batch dimension (b) back to the end
        result = torch.einsum(
            "bijk, bi -> jb", lanczos_vector_stack, weights[:, :n_lanczos]
        )

        return result
