from __future__ import annotations

from logging import getLogger
from typing import Any, Callable

import torch

logger = getLogger(__name__)


def torch_call(
    abstract_fn: str, args: list[str | float]
) -> Callable[[dict], torch.Tensor]:
    """Convert a `Call` object into a torchified function which can be evaluated using
    a vparams and inputs dict.
    """
    fn = getattr(torch, abstract_fn)

    def evaluate(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        tensor_args = []
        for symbol in args:
            if isinstance(symbol, float):
                tensor_args.append(torch.tensor(symbol))
            elif isinstance(symbol, str):
                tensor_args.append(inputs[symbol])
        return fn(*tensor_args)

    return evaluate


class Embedding(torch.nn.Module):
    """
    vparam_names: A list of abstract variational parameter names.
    fparam_names: A list of abstract feature parameter names.
    leaf_to_call: Map from intermediate and leaf variables (which will be used as angles in gates)
                to torch callables which expect a dictionary of root and intermediate variables.
    dtype: The precision of the parameters.
    device: The device on which the parameters are stored.
    """

    def __init__(
        self,
        vparam_names: list[str],
        fparam_names: list[str],
        leaf_to_call: dict[str, Callable],
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.vparams = torch.nn.ParameterDict(
            {p: torch.rand(1, requires_grad=True) for p in vparam_names}
        )
        self.fparams: dict[str, torch.Tensor | None] = {p: None for p in fparam_names}
        self.leaf_to_call: dict[str, Callable] = leaf_to_call
        self._dtype = dtype
        self._device = device
        logger.debug(
            f"Embedding initialized with vparams: {list(self.vparams.keys())},\
                     ,fparams {list(self.fparams.keys())}\
                    and leaf parameters {list(self.leaf_to_call.keys())}."
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

    def flush_fparams(self) -> None:
        for key, _ in self.fparams.items():
            self.fparams[key] = None

    def assign_single_leaf(
        self, leaf_name: str, root_and_intermediates: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {leaf_name: self.leaf_to_call[leaf_name](root_and_intermediates)}

    def assign_leaves(
        self, root_params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Expects a dict of user-passed name:value pairs for featureparameters
        and assigns all intermediate and leaf variables using the current vparam values
        and the passed values for featureparameters."""
        intermediates_and_leaves: dict[str, torch.Tensor] = {}
        try:
            assert root_params.keys() == self.fparams.keys()
        except Exception as e:
            logger.error(
                f"Please pass a dict containing name:value for each fparam. Got {e}"
            )
        for var, torchcall in self.leaf_to_call.items():
            intermediates_and_leaves[var] = torchcall(
                {
                    **self.vparams,
                    **root_params,
                    **intermediates_and_leaves,
                },  # we add the "intermediate" variables too
            )

        return intermediates_and_leaves

    def __call__(self, root_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.assign_leaves(root_params)

    @property
    def root_param_names(self) -> list[str]:
        return list(self.fparams.keys()) + list(self.vparams.keys())

    @property
    def leaf_param_names(self) -> list[str]:
        return list(self.leaf_to_call.keys())

    def set_rootparam(self, param_name: str) -> None:
        # TODO make it possible to make trainable params non-trainable and vice versa
        pass
