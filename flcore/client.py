from __future__ import annotations

import copy
import typing as T
import warnings
from abc import abstractmethod

import torch
import torch.nn as nn

__all__ = ["ClientProtocol", "MetricResult", "Channel"]

Channel = T.NewType("Channel", T.Any)
MetricResult = T.NewType("MetricResult", dict[str, float])


@T.runtime_checkable
class ClientProtocol(T.Protocol):
    """
    The protocol of client. A client is a device that has a local model and can train, evaluate and test the model.
    """

    id: T.Hashable
    device: torch.device
    dataset_size: int
    _context: dict

    @abstractmethod
    def train(self):
        """
        Training the local model.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> MetricResult:
        """
        Evaluating the local model.

        :return: The evaluated result.
        """
        raise NotImplementedError

    @abstractmethod
    def test(self) -> MetricResult:
        """
        Testing the local model.

        :return: The tested result.
        """
        raise NotImplementedError

    @property
    def model(self) -> nn.Module:
        """
        :return: The local model.
        """
        if model := self._context.get("model", None):
            return model
        raise RuntimeError("The model is not received yet.")

    def connect(self) -> Channel:
        """
        Connect to a server.

        :return: The channel to the server.
        """
        self._context = {}
        return Channel(self)

    def close(self):
        if hasattr(self, "_context"):
            del self._context

    def receive_model(self, model: nn.Module):
        """
        Receive a model.

        :param model: The new model.
        """
        self._context["model"] = copy.deepcopy(model).to(self.device)

    def send_model(self) -> nn.Module:
        """
        Send the local model to the server.
        """
        return copy.deepcopy(self.model).cpu()

    def __enter__(self) -> Channel:
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            warnings.warn(
                f"An Exception was raised: {exc_type} - {exc_val}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.close()
