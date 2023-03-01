from __future__ import annotations

import copy
import functools
from typing import Sequence, NamedTuple, Callable, TypeAlias

import torch
import torch.nn as nn

MapFunction: TypeAlias = Callable[..., torch.Tensor]


class ModelInfo(NamedTuple):
    model: nn.Module
    weight: float
    id: str = None


@torch.no_grad()
def compare_model_structures(*models: nn.Module, raise_error: bool = False) -> bool:
    try:
        for params in zip(*[model.parameters() for model in models], strict=True):
            sizes = {param.size() for param in params}
            types = {param.dtype for param in params}

            if len(sizes) != 1:
                raise ValueError(f"Model parameters are not the same size at {params[0]}.")
            if len(types) != 1:
                raise ValueError(f"Model parameters are not the same type at {params[0]}.")

    except ValueError as e:
        if raise_error:
            raise e
        return False

    else:
        return True


@torch.no_grad()
def move_parameters(from_: nn.Module, to: nn.Module, *, buffer: bool = False, zero_grad: bool = False):
    compare_model_structures(from_, to, raise_error=True)

    if buffer:
        to.load_state_dict(from_.state_dict())
    else:
        for from_param, to_param in zip(from_.parameters(), to.parameters()):
            to_param.data.copy_(from_param.data)

    if zero_grad:
        to.zero_grad(set_to_none=True)


@torch.no_grad()
def aggregate_parameters(local_model_infos: Sequence[ModelInfo], global_model: nn.Module) -> nn.Module:
    if len(local_model_infos) == 0:
        raise ValueError("No available model to performs aggregation.")

    compare_model_structures(global_model, *[info.model for info in local_model_infos], raise_error=True)

    for info in local_model_infos:
        local_model = info.model
        weight = info.weight

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            global_param.data.add_(local_param.data - global_param.data, alpha=weight)

    return global_model


@torch.no_grad()
def flatten_model(model: nn.Module) -> torch.Tensor:
    vector = torch.cat([param.flatten() for param in model.parameters()])
    return vector


@torch.no_grad()
def model_map(function: MapFunction, models: Sequence[nn.Module] = None, *, out: nn.Module = None) -> nn.Module:
    if out is None:
        out = copy.deepcopy(models[0])

    compare_model_structures(out, *models, raise_error=True)

    for out_param, *params in zip(out.parameters(), *[model.parameters() for model in models]):
        out_param.data.copy_(function(*[param.data for param in params]))

    return out


@torch.no_grad()
def add(left: nn.Module, right: nn.Module, *, out: nn.Module = None) -> nn.Module:
    return model_map(torch.add, [left, right], out=out)


@torch.no_grad()
def sub(left: nn.Module, right: nn.Module, *, out: nn.Module = None) -> nn.Module:
    return model_map(torch.sub, [left, right], out=out)


@torch.no_grad()
def multiply(x: nn.Module, alpha: float, *, out: nn.Module = None) -> nn.Module:
    mul = functools.partial(torch.mul, alpha)
    return model_map(mul, [x], out=out)


@torch.no_grad()
def scale_delta(x: nn.Module, y: nn.Module, alpha: float, *, out: nn.Module = None) -> nn.Module:
    out = sub(x, y, out=out)
    out = multiply(out, alpha, out=out)
    return out
