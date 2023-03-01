from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn

from . import model as model_utils


class RobustFn(ABC):
    @abstractmethod
    def __call__(
            self, model_infos: Sequence[model_utils.ModelInfo], global_model: nn.Module
    ) -> list[model_utils.ModelInfo]:
        ...

    @staticmethod
    def sum_weight(model_infos: Sequence[model_utils.ModelInfo]):
        return sum([info.weight for info in model_infos])


class Krum(RobustFn):
    def __init__(self, *, num_corrupted: int = 1, num_select: int = 1):
        assert num_corrupted >= 0
        assert num_select > 0
        self.num_corrupted = num_corrupted
        self.num_select = num_select

    @torch.no_grad()
    def __call__(
            self, model_infos: Sequence[model_utils.ModelInfo], global_model: nn.Module
    ) -> list[model_utils.ModelInfo]:
        if not len(model_infos) > 2 * self.num_corrupted + 2:
            warnings.warn(f"The number of aggregated clients is smaller than 2 * f + 2 ({2 * self.num_corrupted + 2}), "
                          f"which not satisfy Krum/MultiKrum need.")

        if len(model_infos) <= self.num_corrupted:
            raise ValueError("Can't select a model when malicious models are more than aggregated models.")

        if len(model_infos) < self.num_select:
            raise ValueError(f"Can's select {self.num_select} models when {len(model_infos)} models only.")

        model_infos = list(model_infos)

        # multi-krum
        selects = []
        for _ in range(self.num_select):
            index = self._krum(model_infos)
            selects.append(model_infos.pop(index))

        # Reset weights
        sum_weight = self.sum_weight(model_infos)
        model_infos = [model_utils.ModelInfo(info.model, 1.0 / sum_weight) for info in selects]

        # Let model_utils.aggregate_parameters() to aggregate
        return model_infos

    @torch.no_grad()
    def _krum(self, model_infos: Sequence[model_utils.ModelInfo]) -> int:
        # Calculate distance between any two models
        vectors = [model_utils.flatten_model(info.model) for info in model_infos]
        distances = [torch.stack([vector.dist(other) for other in vectors]) for vector in vectors]

        # Calculate their scores
        num_select = len(model_infos) - self.num_corrupted - 2
        # torch.sort() return a 2-element tuple, we only need 0th.
        # The 0th is the distance between itself in the sorted distances
        scores = [distance.sort()[0][1:num_select + 1].sum() for distance in distances]

        # Select the minimal
        index = torch.tensor(scores).argmin().item()

        return index


class TrimmedMean(RobustFn):
    def __init__(self, *, num_remove: int):
        assert num_remove >= 0
        self.num_remove = num_remove

    @torch.no_grad()
    def __call__(
            self, model_infos: Sequence[model_utils.ModelInfo], global_model: nn.Module
    ) -> list[model_utils.ModelInfo]:
        if len(model_infos) <= self.num_remove:
            raise ValueError(f"Can't remove {self.num_remove} models when there are {len(model_infos)} models only.")

        global_model = model_utils.model_map(function=self._trimmed_mean, models=[info.model for info in model_infos])
        model_infos = [model_utils.ModelInfo(global_model, self.sum_weight(model_infos))]

        return model_infos

    @torch.no_grad()
    def _trimmed_mean(self, *params: torch.Tensor) -> torch.Tensor:
        # Calculate parameters median and set it to median_model
        stacked_params = torch.stack(params)  # Stack for tensor calculate
        median = stacked_params.median(dim=0)[0]  # Median value along side stack dimension
        median = median.expand(stacked_params.size())  # Propagate to make process clear
        # Select len(model_infos) - self.num_remove models that close to median (element-wise)
        selects = (stacked_params - median).abs().sort(dim=0)[0][: - self.num_remove]
        # Aggregate
        return selects.mean(dim=0)


class Median(RobustFn):
    def __call__(
            self, model_infos: Sequence[model_utils.ModelInfo], global_model: nn.Module
    ) -> list[model_utils.ModelInfo]:
        global_model = model_utils.model_map(function=self._median, models=[info.model for info in model_infos])
        model_infos = [model_utils.ModelInfo(global_model, weight=self.sum_weight(model_infos))]
        return model_infos

    @staticmethod
    def _median(*params: torch.Tensor) -> torch.Tensor:
        median = torch.stack(params).median(dim=0)[0]
        return median


class Bulyan(RobustFn):
    def __init__(self, *, num_corrupted: int = 1):
        self.num_corrupted = num_corrupted

    def __call__(
            self, model_infos: Sequence[model_utils.ModelInfo], global_model: nn.Module
    ) -> list[model_utils.ModelInfo]:
        if not len(model_infos) >= 4 * self.num_corrupted + 3:
            warnings.warn(f"The number of aggregated clients is smaller than 4 * f + 3 ({4 * self.num_corrupted + 3}), "
                          f"which not satisfy Bulyan need.")

        krum = Krum(num_corrupted=self.num_corrupted, num_select=len(model_infos) - 2 * self.num_corrupted)
        trimmed_mean = TrimmedMean(num_remove=2 * self.num_corrupted)

        model_infos = krum(model_infos, global_model)
        model_infos = trimmed_mean(model_infos, global_model)

        return model_infos
