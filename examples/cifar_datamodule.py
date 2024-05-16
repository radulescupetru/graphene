from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from mlx.data.datasets.cifar import load_cifar10

from src.core.datamodule import DataModule


class Cifar10DataModule(DataModule):
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    @staticmethod
    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - Cifar10DataModule.mean) / Cifar10DataModule.std

    def __init__(self, args: SimpleNamespace) -> None:
        super().__init__()
        self.args = args

    def setup(self) -> None:
        self.train_dataset = load_cifar10(root=None, train=True)
        self.valid_dataset = load_cifar10(root=None, train=False)

    def train_dataloader(self):
        if not hasattr(self, "_train_dataloader"):
            self._train_dataloader = (
                self.train_dataset.shuffle()
                .to_stream()
                .image_random_h_flip("image", prob=0.5)
                .pad("image", 0, 4, 4, 0.0)
                .pad("image", 1, 4, 4, 0.0)
                .image_random_crop("image", 32, 32)
                .key_transform("image", Cifar10DataModule.normalize)
                .batch(self.args.batch_size)
                .prefetch(4, 4)
            )
        return self._train_dataloader

    def valid_dataloader(self):
        if not hasattr(self, "_valid_dataloader"):
            self._valid_dataloader = (
                self.valid_dataset.to_stream()
                .key_transform("image", Cifar10DataModule.normalize)
                .batch(self.args.batch_size)
            )
        return self._valid_dataloader

    def test_dataloader(self):
        return self.valid_dataloader()
