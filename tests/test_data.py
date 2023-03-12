import pytest
import torch.utils.data as data
from torchvision.datasets import CIFAR100, EMNIST

import flcore.utils.data as data_utils

DATA_DIR = "data"


@pytest.fixture(scope="module")
def cifar100():
    dataset = data.ConcatDataset([
        CIFAR100(root=DATA_DIR, download=True, train=True),
        CIFAR100(root=DATA_DIR, download=True, train=False),
    ])
    return dataset


@pytest.fixture(scope="module")
def emnist():
    dataset = data.ConcatDataset([
        EMNIST(root=DATA_DIR, split="byclass", download=True, train=True),
        EMNIST(root=DATA_DIR, split="byclass", download=True, train=False),
    ])
    return dataset


@pytest.fixture
def datasets(cifar100, emnist):
    return [cifar100, emnist]


def test_default_get_target(datasets):
    for dataset in datasets:
        item = dataset[0]
        assert data_utils.default_get_target(item) == item[-1]


def test_get_targets(datasets):
    for dataset in datasets:
        targets = data_utils.get_targets(dataset=dataset)

        for item, target in zip(dataset, targets):
            assert target == item[-1]


def test_collect_targets(datasets):
    for dataset in datasets:
        targets = data_utils.get_targets(dataset=dataset)
        target_indexes = data_utils.collect_targets(targets)

        assert set(target_indexes) == set(targets)
        assert sum(map(len, target_indexes.values())) == len(targets)


def test_random_split_both_lengths_and_fractions(datasets):
    for dataset in datasets:
        targets = data_utils.get_targets(dataset=dataset)
        length = len(targets)
        with pytest.raises(ValueError):
            data_utils.random_split(targets, lengths=[length // 2, length - (length // 2)], fractions=[0.5, 0.5])


def test_random_split_no_lengths_and_fractions(datasets):
    for dataset in datasets:
        targets = data_utils.get_targets(dataset=dataset)

        with pytest.raises(ValueError):
            data_utils.random_split(targets)


def test_random_split_wrong_lengths(datasets):
    for dataset in datasets:
        targets = data_utils.get_targets(dataset=dataset)

        with pytest.raises(ValueError):
            data_utils.random_split(targets, lengths=[100, 1])


def test_random_split_wrong_fractions(datasets):
    for dataset in datasets:
        targets = data_utils.get_targets(dataset=dataset)

        with pytest.raises(ValueError):
            data_utils.random_split(targets, fractions=[1.0, 0.001])

        with pytest.raises(ValueError):
            data_utils.random_split(targets, fractions=[0.5, 0.1, 0.1])

        with pytest.raises(ValueError):
            data_utils.random_split(targets, fractions=[-0.5, 0.5, 1.0])


def test_random_split_length(datasets):
    for dataset in datasets:
        targets = data_utils.get_targets(dataset=dataset)
        fractions = [1 / len(targets)] * len(targets)
        split = data_utils.random_split(targets, fractions=fractions)

        for value in split:
            assert len(value) == 1


def test_partition_dataset_wrong_partition_func(datasets):
    for dataset in datasets:
        with pytest.raises(ValueError):
            data_utils.partition_dataset(dataset, num_partition=1, partition_func=lambda: [0.5, 0.5])


def test_partition_dataset(datasets):
    for dataset in datasets:
        subsets = data_utils.partition_dataset(dataset,
                                               num_partition=100,
                                               partition_func=lambda: [1 / 100] * 100)
        assert sum(map(len, subsets)) == len(dataset)


def test_generate_dirichlet_subsets(datasets):
    for dataset in datasets:
        subsets = data_utils.generate_dirichlet_subsets(dataset, alphas=[5] * 100)
        assert sum(map(len, subsets)) == len(dataset)


def test_generate_iid_subsets(datasets):
    for dataset in datasets:
        subsets = data_utils.generate_iid_subsets(dataset, num_partition=100)

        assert sum(map(len, subsets)) == len(dataset)

        targets = set(data_utils.get_targets(dataset))
        for subset in subsets:
            assert set(data_utils.get_targets(subset)) == targets
