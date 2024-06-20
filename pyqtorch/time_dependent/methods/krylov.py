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

        def exponentiate() -> tuple[torch.Tensor, bool]:
            # approximate next iteration by modifying T, and unmodifying
            T[i - 1, i] = 0
            T[i + 1, i] = 1
            exp = torch.linalg.matrix_exp(-1j * dt * T[: i + 2, : i + 2].clone())
            T[i - 1, i] = T[i, i - 1]
            T[i + 1, i] = 0

            e1 = abs(exp[i, 0])
            e2 = abs(exp[i + 1, 0]) * n
            if e1 > 10 * e2:
                error = e2
            elif e2 > e1:
                error = e1
            else:
                error = (e1 * e2) / (e1 - e2)

            converged = error < self.options.exp_tolerance
            return exp[:, 0], converged

        lanczos_vectors = [state]
        T = torch.zeros(
            self.options.max_krylov + 1, self.options.max_krylov + 1, dtype=state.dtype
        )

        # step 0 of the loop
        v = torch.matmul(self.H(t), state)
        a = torch.matmul(v.conj().T, state)
        n = torch.linalg.vector_norm(v)
        T[0, 0] = a
        v = v - a * state

        for i in range(1, self.options.max_krylov):
            # this block should not be executed in step 0
            b = torch.linalg.vector_norm(v)
            if b < self.options.norm_tolerance:
                exp = torch.linalg.matrix_exp(-1j * dt * T[:i, :i])
                weights = exp[:, 0]
                converged = True
                break

            T[i, i - 1] = b
            T[i - 1, i] = b
            state = v / b
            lanczos_vectors.append(state)
            weights, converged = exponentiate()
            if converged:
                break

            v = torch.matmul(self.H(t), state)
            a = torch.matmul(v.conj().T, state)
            n = torch.linalg.vector_norm(v)
            T[i, i] = a
            v = v - a * lanczos_vectors[i] - b * lanczos_vectors[i - 1]

        if not converged:
            raise RecursionError(
                "Exponentiation algorithm did not converge \
                to precision in allotted number of steps."
            )

        result = lanczos_vectors[0] * weights[0]
        for i in range(1, len(lanczos_vectors)):
            result += lanczos_vectors[i] * weights[i]

        return result
