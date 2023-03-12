from __future__ import annotations

import contextlib
import pathlib
from typing import Any

import torch


def dump(obj: Any, filename: str | pathlib.Path, *, replace: bool = False):
    filename = pathlib.Path(filename)
    temp_file = filename.with_suffix("tmp")
    torch.save(obj, temp_file)

    if replace:
        temp_file.replace(filename)
    else:
        temp_file.rename(filename)


def load(filename: str | pathlib.Path, *, raise_error: bool = True) -> Any:
    cm = contextlib.suppress(RuntimeError) if raise_error else contextlib.nullcontext()

    with cm:
        return torch.load(filename)
