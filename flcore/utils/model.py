from __future__ import annotations

import copy
from typing import Sequence, Callable, Optional, Iterable

import torch
import torch.nn as nn


@torch.no_grad()
def move_parameters(from_: nn.Module, to: nn.Module, *, buffer: bool = False, zero_grad: bool = False):
    if buffer:
        to.load_state_dict(from_.state_dict())
    else:
        for from_param, to_param in zip(from_.parameters(), to.parameters(), strict=True):
            to_param.data.copy_(from_param.data)

    if zero_grad:
        to.zero_grad(set_to_none=True)


@torch.no_grad()
def model_to_vector(model: nn.Module) -> torch.Tensor:
    # Flag for the device where the parameter is located
    param_device = None

    vector = []
    for param in model.parameters():
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vector.append(param.view(-1))

    return torch.cat(vector)


def vector_to_model(vector: torch.Tensor, model: nn.Module) -> None:
    # Ensure vec of type Tensor
    if not isinstance(vector, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, but got: {torch.typename(vector)}")

    # Flag for the device where the parameter is located
    param_device = None
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in model.parameters():
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()

        # Slice the vector, reshape it, and copy the old data of the parameter
        param.data.copy_(vector[pointer: pointer + num_param].view_as(param).data)

        # Increment the pointer
        pointer += num_param


@torch.no_grad()
def aggregate_gradient(global_vector: torch.Tensor, local_grads: Iterable[torch.Tensor], weights: Iterable[float], *,
                       out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = global_vector.clone()
    else:
        out.copy_(global_vector)

    for weight, local_grad in zip(weights, local_grads, strict=True):
        out.add_(local_grad, alpha=weight)

    return out


@torch.no_grad()
def aggregate(global_vector: torch.Tensor, local_vectors: Iterable[torch.Tensor], weights: Iterable[float], *,
              out: torch.Tensor = None) -> torch.Tensor:
    local_grads = (vector - global_vector for vector in local_vectors)
    return aggregate_gradient(global_vector, local_grads, weights, out=out)


@torch.no_grad()
def aggregate_model(global_model: nn.Module, local_models: Iterable[nn.Module], weights: Iterable[float]) -> nn.Module:
    global_vector = model_to_vector(global_model)
    local_vectors = (model_to_vector(local_model) for local_model in local_models)

    global_vector = aggregate(global_vector, local_vectors, weights, out=global_vector)
    vector_to_model(global_vector, global_model)

    return global_model


@torch.no_grad()
def layer_map(
        function: Callable[[tuple[torch.Tensor, ...]], torch.Tensor], models: Sequence[nn.Module], *,
        out: nn.Module = None
) -> nn.Module:
    out = out or copy.deepcopy(models[0])

    for out_param, *params in zip(out.parameters(), *[model.parameters() for model in models]):
        params = tuple([param.data for param in params])
        result = function(params)
        out_param.data.copy_(result)

    return out


# From PyTorch
def _check_param_device(param: torch.Tensor, old_param_device: Optional[int]) -> int:
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        if param.is_cuda:  # Check if in same GPU
            warn = (param.get_device() != old_param_device)
        else:  # Check if in CPU
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, '
                            'this is currently not supported.')
    return old_param_device
