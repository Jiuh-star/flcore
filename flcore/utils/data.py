# REFERENCE: Federated Learning on Non-IID Data Silos: An Experimental Study
from __future__ import annotations

import math
import random
from typing import Callable, Any, Sized, TypeVar, Sequence, TypeAlias

import more_itertools
import torch
import torch.distributions as distributions
import torch.utils.data as data

T = TypeVar("T")
Feature = TypeVar("Feature")
Target = TypeVar("Target")
Dataset: TypeAlias = data.Dataset | Sized


def default_get_target(item: tuple[Feature, Target]) -> Target:
    """
    Default function that receives an item from dataset, and then return its target.

    :param item: An item from dataset.
    :return: Target of the item.
    """
    return item[-1]


def collect_targets(targets: Sequence[Target]) -> dict[Target, list[Target]]:
    """
    Collect and classify the targets to a dict. In dict, the keys are targets and value are corresponding indexes.

    :param targets: A sequence of target.
    :return: Collected targets.
    """
    target_indexes = {}
    for index, target in enumerate(targets):
        target_indexes.setdefault(target, []).append(index)
    return target_indexes


def get_targets(dataset: Dataset,
                get_target: Callable[[Any], Target] = default_get_target) -> list[Target]:
    """
    Get targets from dataset.

    :param dataset: The dataset.
    :param get_target: A function that receives an item from dataset and then return its target.
    :return: A list of targets of dataset.
    """
    # dataset from torchvision has attribute targets
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        # some targets are Tensor type
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()

        return list(targets)

    # some user may concat torchvision dataset
    if isinstance(dataset, data.ConcatDataset):
        targets = []
        for _dataset in dataset.datasets:
            targets.extend(get_targets(_dataset))
        return targets

    targets = [get_target(item) for item in dataset]
    return targets


def random_split(values: Sequence[T],
                 lengths: Sequence[int | float] = None,
                 fractions: Sequence[float] = None) -> list[list[T]]:
    """
    Randomly split *values* into some list of value.

    :param values: A Sequence of values to be split.
    :param lengths: Lengths of subsequence. Should equals to len(values).
    :param fractions: Fractions of subsequence length. Should be closed to 1 enough.
    :return: A list of sublist of values.
    """
    if lengths and fractions:
        raise ValueError("Both lengths and fractions are specified, consider remove one.")

    if not lengths and not fractions:
        raise ValueError("No values in lengths and fractions at the same time.")

    if lengths and sum(lengths) != len(values):
        raise ValueError("Sum of lengths is not equals to values length.")

    if fractions:
        if not math.isclose(sum(fractions), 1., abs_tol=1E-5):
            raise ValueError("Sum of fractions is not close enough to 1.")

        # fractions to lengths
        lengths = []
        for i, frac in enumerate(fractions):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            length = math.floor(len(values) * frac)
            lengths.append(length)

        # sum(lengths) may not exactly equals to len(values)
        remainder = len(values) - sum(lengths)
        remainder = remainder if remainder >= 0 else - remainder
        step = 1 if remainder >= 0 else -1

        # put remainder into lengths
        for i in range(remainder):
            lengths[i % len(lengths)] += step

    values = list(values)
    random.shuffle(values)
    values = list(more_itertools.split_into(values, lengths))

    return values


def partition_dataset(dataset: Dataset,
                      num_partition: int,
                      partition_func: Callable[[], list[float]],
                      get_target: Callable[[Any], Target] = default_get_target) -> list[data.Subset, ...]:
    targets = get_targets(dataset, get_target)
    target_indexes = collect_targets(targets)
    target_split_indexes = {}

    for target in target_indexes:
        fractions = partition_func()
        target_split_indexes[target] = random_split(target_indexes[target], fractions=fractions)

    # make subsets
    subsets = []
    for _ in range(num_partition):  # for each partition
        indexes = []
        for target in target_split_indexes:  # put split indexes of each class to a partition
            indexes.extend(target_split_indexes[target].pop())

        subset = data.Subset(dataset, indexes)
        subsets.append(subset)

    return subsets


def generate_dirichlet_subsets(dataset: Dataset,
                               alphas: Sequence[float],
                               get_target: Callable[[Any], Target] = default_get_target,
                               min_data: int = 10,
                               max_retry: int = 10) -> list[data.Subset, ...]:
    def dirichlet() -> list[float]:
        m = distributions.Dirichlet(torch.tensor(alphas, dtype=torch.float))
        sample = m.sample().tolist()
        return sample

    # verify min_data was satisfied
    for _ in range(max_retry):
        subsets = partition_dataset(dataset, num_partition=len(alphas), partition_func=dirichlet, get_target=get_target)
        if all([len(subset) >= min_data for subset in subsets]):
            break
    else:
        raise TimeoutError(
            f"Unable to sample from dirichlet distributions (α = {alphas}) within {max_retry} retries "
            f"to satisfy that each client holds {min_data} data at least."
        )

    return subsets


def generate_iid_subsets(dataset: Dataset,
                         num_client: int,
                         get_target: Callable[[Any], Target] = default_get_target) -> list[data.Subset, ...]:
    return partition_dataset(dataset, num_partition=num_client, partition_func=lambda: [1 / num_client] * num_client,
                             get_target=get_target)
