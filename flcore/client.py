from __future__ import annotations

from abc import abstractmethod
from typing import NewType, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from .utils import model as model_utils

MetricResult = NewType("MetricResult", dict[str, float])


@runtime_checkable
class ClientProtocol(Protocol):
    """
    The protocol of client. A client is a device that has a local model and can train, evaluate and test the model.
    """
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
        """
        A solution of `__init__()` problem when subclass `Protocol`. This method set up client's attributes.

        :param id_:  Identification of the client.
        :param model: The model of client, namely a local model.
        :param learning_rate: The learning rate of local model.
        :param max_epoch: The max epoch of local model training. Typically set to 1.
        :param train_dataloader: The dataloader for model train.
        :param eval_dataloader: The dataloader for model evaluation.
        :param test_dataloader: The dataloader for model test.
        :param device: Which device does the local model used. Typically set to cuda.
        :param optimizer: The optimizer of local model.
        :param loss_fn: The loss function of local model.
        """
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
        """
        Training the local model with `dataloader`.

        :param dataloader: The dataloader for training.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, dataloader: data.DataLoader) -> MetricResult:
        """
        Evaluating the local model with `dataloader`.

        :param dataloader: The dataloader for evaluation.
        :return: The evaluated result.
        """
        raise NotImplementedError

    @property
    def train_dataset(self):
        """ The training dataset. """
        return self.train_dataloader.dataset

    @property
    def eval_dataset(self):
        """ The evaluation dataset. """
        return self.eval_dataloader.dataset

    @property
    def test_dataset(self):
        """ The test dataset. """
        return self.test_dataloader.dataset

    def receive_model(self, new_model: nn.Module):
        """
        Receive a new model, which typically, a global model. The method make sure the optimizer of the client tracks
        the valid model parameters.

        :param new_model: The new model.
        """
        model_utils.move_parameters(new_model, self.model, zero_grad=True)
