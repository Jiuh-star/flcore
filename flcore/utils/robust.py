from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Sequence, TypeAlias

import torch
import torch.nn as nn

from . import model as model_utils

RobustFnReturn: TypeAlias = tuple[list[nn.Module], list[float] | None]


class RobustFn(ABC):
    @abstractmethod
    def __call__(
            self, global_model: nn.Module, local_models: Sequence[nn.Module], weights: Sequence[float] = None
    ) -> RobustFnReturn:
        ...


class Krum(RobustFn):
    def __init__(self, *, num_remove: int, num_select: int):
        assert num_remove > 0
        assert num_select > 0
        self.num_remove = num_remove
        self.num_select = num_select

    def __call__(
            self, global_model: nn.Module, local_models: Sequence[nn.Module], weights: Sequence[float] = None
    ) -> RobustFnReturn:
        if not len(local_models) - (2 * self.num_remove + 2) >= self.num_select:
            warnings.warn(f"The number of aggregated clients is smaller than 2 * f + 2 ({2 * self.num_remove + 2}), "
                          f"which not satisfy Krum/MultiKrum need.")

        if len(local_models) < self.num_remove + self.num_select:
            raise ValueError(f"Can't select {self.num_select} models and remove {self.num_remove} models "
                             f"when there are {len(local_models)} models only.")

        local_models = list(local_models)
        weights = list(weights) if weights else [1 / len(local_models)] * len(local_models)
        local_vectors = [model_utils.model_to_vector(model) for model in local_models]

        # multi-krum
        selects = []
        select_weights = []
        for _ in range(self.num_select):
            index = self._krum(local_vectors)
            local_vectors.pop(index)
            selects.append(local_models.pop(index))
            select_weights.append(weights.pop(index))

        return selects, select_weights

    def _krum(self, vectors: Sequence[torch.Tensor]) -> int:
        # Calculate distance between any two vectors
        distances = [torch.cat([vector.dist(other) for other in vectors]) for vector in vectors]

        # Calculate their scores
        num_select = len(vectors) - self.num_remove - 2
        # torch.sort() return a 2-element tuple, we only need 0th.
        # The 0th is the distance between itself in the sorted distances
        scores = [distance.sort()[0][1:num_select + 1].sum() for distance in distances]

        # Select the minimal
        index = torch.tensor(scores).argmin().item()

        return index


class TrimmedMean(RobustFn):
    def __init__(self, *, num_remove: int):
        assert num_remove > 0
        self.num_remove = num_remove

    def __call__(
            self, global_model: nn.Module, local_models: Sequence[nn.Module], weights: Sequence[float] = None
    ) -> RobustFnReturn:
        if len(local_models) <= self.num_remove:
            raise ValueError(f"Can't remove {self.num_remove} models when there are {len(local_models)} models only.")

        remove_left = self.num_remove // 2
        remove_right = self.num_remove - remove_left

        local_vectors = [model_utils.model_to_vector(model) for model in local_models]
        trimmed_mean = torch.stack(local_vectors).sort(dim=0)[0][remove_left: -remove_right].mean(dim=0)

        global_model = copy.deepcopy(global_model)
        model_utils.vector_to_model(trimmed_mean, global_model)

        return [global_model], None


class Median(RobustFn):
    def __call__(
            self, global_model: nn.Module, local_models: Sequence[nn.Module], weights: Sequence[float] = None
    ) -> RobustFnReturn:
        local_vectors = [model_utils.model_to_vector(model) for model in local_models]
        median = torch.stack(local_vectors).median(dim=0)[0]

        global_model = copy.deepcopy(global_model)
        model_utils.vector_to_model(median, global_model)

        return [global_model], None


class Bulyan(RobustFn):
    def __init__(self, *, num_remove: int):
        assert num_remove > 0
        self.num_remove = num_remove

        self.krum = None
        self.trimmed_mean = None

    def __call__(
            self, global_model: nn.Module, local_models: Sequence[nn.Module], weights: Sequence[float] = None
    ) -> RobustFnReturn:
        if not len(local_models) >= 4 * self.num_remove + 3:
            warnings.warn(f"The number of aggregated clients is smaller than 4 * f + 3 ({4 * self.num_remove + 3}), "
                          f"which not satisfy Bulyan need.")

        if self.krum is None:
            self.krum = Krum(num_remove=self.num_remove, num_select=2 * self.num_remove + 3)

        if self.trimmed_mean is None:
            self.trimmed_mean = TrimmedMean(num_remove=2 * self.num_remove)

        local_models, weights = self.krum(global_model, local_models, weights)
        local_models, weights = self.trimmed_mean(global_model, local_models, weights)

        return local_models, weights


class NormBound(RobustFn):
    def __init__(self, *, threshold: float):
        self.threshold = threshold

    def __call__(
            self, global_model: nn.Module, local_models: Sequence[nn.Module], weights: Sequence[float] = None
    ) -> RobustFnReturn:
        global_vector = model_utils.model_to_vector(global_model)

        models = []
        for model in local_models:
            vector = model_utils.model_to_vector(model)
            grad = vector - global_vector
            grad = grad / max(1, grad.norm() / self.threshold)

            model = copy.deepcopy(model)
            model_utils.vector_to_model(global_vector + grad, model)
            models.append(model)

        return models, list(weights) if weights else None
