from __future__ import annotations

import copy
import math
import typing as T
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from . import model as model_utils

RobustFnReturn: T.TypeAlias = tuple[list[nn.Module], list[float]]


class RobustFn(ABC):
    """
    The abstract class of the robust aggregation function.
    """

    @abstractmethod
    def __call__(
            self,
            global_model: nn.Module,
            local_models: T.Sequence[nn.Module],
            weights: T.Sequence[float],
    ) -> RobustFnReturn:
        ...


class Krum(RobustFn):
    def __init__(self, *, num_remove: int):
        """
        The Krum robust aggregation function.

        :param num_remove: How many models to remove.
        """
        assert num_remove > 0
        self.num_remove = num_remove

    def __call__(
            self,
            global_model: nn.Module,
            local_models: T.Sequence[nn.Module],
            weights: T.Sequence[float],
    ) -> RobustFnReturn:
        if self.num_remove >= len(local_models):
            raise ValueError(
                f"Can not remove {self.num_remove} models when there are only {len(local_models)} models."
            )

        if 2 * self.num_remove + 2 >= len(local_models):
            warnings.warn(
                f"There are only {len(local_models)} models, "
                f"which not satisfy Krum/MultiKrum needs {2 * self.num_remove + 2}."
            )

        local_models = list(local_models)
        weights = list(weights)
        local_vectors = [model_utils.model_to_vector(model) for model in local_models]

        # multi-krum
        num_select = len(local_models) - self.num_remove
        selected_models = []
        selected_weights = []
        for _ in range(num_select):
            index = self._krum(local_vectors)
            local_vectors.pop(index)
            selected_models.append(local_models.pop(index))
            selected_weights.append(weights.pop(index))

        return selected_models, selected_weights

    def _krum(self, vectors: T.Sequence[torch.Tensor]) -> int:
        # Calculate distance between any two vectors
        distances = [
            torch.stack([vector.dist(other) for other in vectors]) for vector in vectors
        ]

        # Calculate their scores
        num_select = len(vectors) - self.num_remove - 2
        # torch.sort() return a 2-element tuple, we only need 0th.
        # The 0th is the distance between itself in the sorted distances
        scores = [
            distance.sort()[0][1: num_select + 1].sum() for distance in distances
        ]

        # Select the minimal
        index = int(torch.tensor(scores).argmin().item())

        return index


class TrimmedMean(RobustFn):
    def __init__(self, *, num_remove: int):
        """
        The TrimmedMean robust aggregation function.

        :param num_remove: How many models to remove, this will remove num_remove // 2 models from left and right.
        """
        assert num_remove > 0
        self.num_remove = num_remove

    def __call__(
            self,
            global_model: nn.Module,
            local_models: T.Sequence[nn.Module],
            weights: T.Sequence[float],
    ) -> RobustFnReturn:
        if len(local_models) <= self.num_remove:
            raise ValueError(
                f"Can't remove {self.num_remove} models when there are {len(local_models)} models only."
            )

        remove_left = self.num_remove // 2
        remove_right = self.num_remove - remove_left

        local_vectors = [model_utils.model_to_vector(model) for model in local_models]
        trimmed_mean = (
            torch.stack(local_vectors)
            .sort(dim=0)[0][remove_left:-remove_right]
            .mean(dim=0)
        )

        robust_model = copy.deepcopy(global_model)
        model_utils.vector_to_model(trimmed_mean, robust_model)

        return [robust_model], [math.fsum(weights)]


class Median(RobustFn):
    def __init__(self):
        """
        The Median robust aggregation function.
        """
        super().__init__()

    def __call__(
            self,
            global_model: nn.Module,
            local_models: T.Sequence[nn.Module],
            weights: T.Sequence[float],
    ) -> RobustFnReturn:
        local_vectors = [model_utils.model_to_vector(model) for model in local_models]
        median = torch.stack(local_vectors).median(dim=0)[0]

        robust_model = copy.deepcopy(global_model)
        model_utils.vector_to_model(median, robust_model)

        return [robust_model], [math.fsum(weights)]


class Bulyan(RobustFn):
    def __init__(self, *, num_remove: int):
        """
        The Bulyan robust aggregation function.

        :param num_remove: How many models to remove.
        """
        assert num_remove > 0
        self.num_remove = num_remove

        self.krum = None
        self.trimmed_mean = None

    def __call__(
            self,
            global_model: nn.Module,
            local_models: T.Sequence[nn.Module],
            weights: T.Sequence[float],
    ) -> RobustFnReturn:
        if not len(local_models) >= 4 * self.num_remove + 3:
            warnings.warn(
                f"The number of aggregated clients is smaller than 4 * f + 3 ({4 * self.num_remove + 3}), "
                f"which not satisfy Bulyan needs."
            )

        if self.krum is None:
            self.krum = Krum(num_remove=2 * self.num_remove)

        if self.trimmed_mean is None:
            self.trimmed_mean = TrimmedMean(num_remove=2 * self.num_remove)

        local_models, weights = self.krum(global_model, local_models, weights)
        local_models, weights = self.trimmed_mean(global_model, local_models, weights)

        return local_models, weights


class NormBound(RobustFn):
    def __init__(self, *, threshold: float):
        """
        The NormBound robust aggregation function.

        :param threshold: The threshold of the norm of the gradient.
        """
        assert threshold > 0
        self.threshold = threshold

    def __call__(
            self,
            global_model: nn.Module,
            local_models: T.Sequence[nn.Module],
            weights: T.Sequence[float],
    ) -> RobustFnReturn:
        global_vector = model_utils.model_to_vector(global_model)

        models = []
        for model in local_models:
            vector = model_utils.model_to_vector(model)
            update = vector - global_vector
            update = update / max(1, update.norm() / self.threshold)

            model = copy.deepcopy(model)
            model_utils.vector_to_model(global_vector + update, model)
            models.append(model)

        return models, list(weights)
