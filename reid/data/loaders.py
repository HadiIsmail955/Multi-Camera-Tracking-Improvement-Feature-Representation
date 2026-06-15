from torch.utils.data import DataLoader

from reid.data.datasets import ReIDDataset
from reid.data.transforms import (
    build_train_transform,
    build_eval_transform,
)
from reid.data.samplers import PKSampler


def build_train_loader(
    records,
    P,
    K,
    num_workers=4,
):
    dataset = ReIDDataset(
        records=records,
        transform=build_train_transform(),
        relabel=True,
    )

    sampler = PKSampler(
        pid_to_indices=dataset.pid_to_indices,
        P=P,
        K=K,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, loader


def build_eval_loader(
    records,
    batch_size=64,
    num_workers=4,
):
    dataset = ReIDDataset(
        records,
        build_eval_transform(),
        relabel=False,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, loader