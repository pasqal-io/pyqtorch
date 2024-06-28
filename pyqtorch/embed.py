from __future__ import annotations

from logging import getLogger
from typing import Any, Callable

import torch

logger = getLogger(__name__)


class ParameterBuffer(torch.nn.Module):
    """A class holding all root parameters either passed by the user
    or trainable variational parameters."""

    def __init__(
        self,
        trainable_vars: list[str],
        non_trainable_vars: list[str],
    ) -> None:
        super().__init__()
        self.vparams = {p: torch.rand(1, requires_grad=True) for p in trainable_vars}
        self.fparams = {p: None for p in non_trainable_vars}
        self._dtype = torch.float64
        self._device = torch.device("cpu")
        logger.debug(
            f"ParameterBuffer initialized with trainable parameters: {self.vparams.keys()},\
                     and non-trainable parameters {self.fparams.keys()}."
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, args: Any, kwargs: Any) -> None:
        self.vparams = {p: t.to(*args, **kwargs) for p, t in self.vparams.items()}
        try:
            k = next(iter(self.vparams))
            t = self.vparams[k]
            self._device = t.device
            self._dtype = t.dtype
        except Exception:
            pass

    def assign_fparams(self, inputs: dict[str, torch.Tensor]) -> None:
        for key, _ in self.fparams.items():
            self.fparams[key] = inputs[key]


class Embedding(torch.nn.Module):
    """A class holding:
    - A parameterbuffer (containing concretized vparams + list of featureparams,
    - A dictionary of intermediate and leaf variable names mapped to a TorchCall object
        which can be results of function/expression evaluations.
    """

    def __init__(self, param_buffer, var_to_call: dict[str, Callable]) -> None:
        super().__init__()
        self.param_buffer = param_buffer
        self.var_to_call: dict[str, Callable] = var_to_call

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expects a dict of user-passed name:value pairs for featureparameters
        and assigns all intermediate and leaf variables using the current vparam values
        and the passed values for featureparameters."""
        assigned_params: dict[str, torch.Tensor] = {}
        try:
            assert inputs.keys() == self.param_buffer.fparams.keys()
        except Exception as e:
            logger.error(
                f"Please pass a dict containing name:value for each fparam. Got {e}"
            )
        for var, torchcall in self.var_to_call.items():
            assigned_params[var] = torchcall(
                self.param_buffer.vparams,
                {
                    **inputs,
                    **assigned_params,
                },  # we add the "intermediate" variables too
            )

        return assigned_params
