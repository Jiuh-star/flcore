from __future__ import annotations

import pathlib
from abc import abstractmethod
from typing import NewType, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from .utils import io
from .utils import model as model_utils

MetricResult = NewType("MetricResult", dict[str, float])


@runtime_checkable
class ClientProtocol(Protocol):
    id: str
    model: nn.Module
    learning_rate: float
    max_epoch: int
    device: torch.device
    train_dataloader: data.DataLoader
    eval_dataloader: data.DataLoader
    test_dataloader: data.DataLoader
    optimizer: optim.Optimizer
    loss_fn: nn.Module

    def setup(self, *, id_: str, model: nn.Module, learning_rate: float, max_epoch: int,
              train_dataloader: data.DataLoader, eval_dataloader: data.DataLoader, test_dataloader: data.DataLoader,
              device: torch.device, optimizer: optim.Optimizer, loss_fn: nn.Module):
        self.id = id_
        self.model = model
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @abstractmethod
    def train(self, dataloader: data.DataLoader):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, dataloader: data.DataLoader) -> MetricResult:
        raise NotImplementedError

    @property
    def train_dataset(self):
        return self.train_dataloader.dataset

    @property
    def eval_dataset(self):
        return self.eval_dataloader.dataset

    @property
    def test_dataset(self):
        return self.test_dataloader.dataset

    def receive_model(self, new_model: nn.Module):
        model_utils.move_parameters(new_model, self.model, zero_grad=True)


class LowMemoryClientMixin:
    def __init__(self, *, state_path: str | pathlib.Path):
        self.state_path = state_path.with_suffix(".lm")
        io.dump((None, None), self.state_path, replace=True)

    @property
    def model(self):
        model, optimizer = io.load(self.state_path)
        return model

    @model.setter
    def model(self, new_model):
        state = (new_model, self.optimizer)
        io.dump(state, self.state_path, replace=True)

    @property
    def optimizer(self):
        model, optimizer = io.load(self.state_path)
        return optimizer

    @optimizer.setter
    def optimizer(self, new_optim):
        state = (self.model, new_optim)
        io.dump(state, self.state_path, replace=True)
