from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.optimizers as optim
from mlx import nn

from examples.cifar_datamodule import Cifar10DataModule
from examples.resnet import resnet20
from src.core.trainer import Trainer
from src.core.trainmodule import TrainModule
from src.metrics import Accuracy


class CifarTrainModule(TrainModule):
    def __init__(self, args: SimpleNamespace) -> None:
        super().__init__(args)
        self.model = resnet20()

    def configure_optimizers(self):
        return optim.Adam(learning_rate=1e-3)

    def forward(self, x) -> Any:
        return self.model(x)

    def setup(self):
        self.accs = []
        self.validation_accuracy = Accuracy()
        self.train_accuracy = Accuracy()

    def training_step(self, batch: dict, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)
        loss = nn.losses.cross_entropy(y_hat, y, reduction="mean")
        self.train_accuracy(y_hat, y)
        self.log("train_accuracy", self.train_accuracy)
        return loss

    def validation_step(self, batch: dict, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.forward(x)
        loss = nn.losses.cross_entropy(y_hat, y, reduction="mean")
        self.validation_accuracy(y_hat, y)
        self.log("validation_accuracy", self.validation_accuracy)
        return loss


if __name__ == "__main__":
    datamodule = Cifar10DataModule(args=SimpleNamespace(batch_size=256))
    trainmodule = CifarTrainModule(args=None)
    trainer = Trainer(train_module=trainmodule, data_module=datamodule, max_epochs=30, run_validation_every_n_epochs=1)
    trainer.fit()
