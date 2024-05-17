from __future__ import annotations

import mlx.core as mx
import numpy as np

from src.core.datamodule import DataModule
from src.core.trainmodule import TrainModule
from src.loops.loop import Loop
from src.metrics.loss import LossMetric
from src.metrics.metric import Metric


class ValidationLoop(Loop):
    def __init__(self, train_module: TrainModule, data_module: DataModule) -> None:
        super().__init__(train_module, data_module)
        self.metrics: dict[str, Metric] = {"loss": LossMetric()}

    def log(self, name, value):
        self.metrics[name] = value

    def on_validation_epoch_start(self):
        """Validation hook which triggers at the beginning of a validation epoch The essential bit
        is setting the model in validation mode.

        Additionally it:
            - calls the user defined method with the same signature.
            - calls the system and user defined callbacks methods with the same signature.
        """
        # Set the model to val mode at the start of the validation epoch
        self.model.train(False)
        # Call user defined method
        if hasattr(self.train_module, "on_validation_epoch_start"):
            self.train_module.on_validation_epoch_start()
        for metric in self.metrics.values():
            metric.reset()

    def on_validation_batch_start(self, batch: np.array, batch_idx: int) -> mx.array:
        """Validation hook which triggers at the start of a validation batch.

        Args:
            batch : A batch of data
            batch_idx: Index of the batch

        Returns:
            batch: Batch formatted as a mx.array on top of additional user defined transformations.
        """
        # Call user defined method
        if hasattr(self.train_module, "on_validation_batch_start"):
            batch = self.train_module.on_validation_batch_start(batch, batch_idx)
        batch = {k: mx.array(v) for k, v in batch.items()}
        return batch

    def on_validation_batch_end(self, loss):
        # Call user defined method
        if hasattr(self.train_module, "on_validation_batch_end"):
            self.train_module.on_validation_batch_start(loss)
        self.metrics["loss"].update(loss)

    def validation_step(self, batch, batch_idx) -> tuple[mx.array, mx.array]:
        assert hasattr(self.train_module, "validation_step"), "The trainmodule has not validation_step defined"
        loss = self.train_module.validation_step(batch, batch_idx)
        return loss

    def on_validation_epoch_end(self):
        # Call user defined method
        if hasattr(self.train_module, "on_validation_epoch_end"):
            self.train_module.on_validation_epoch_end()
        # Reset the data stream
        self.data_module.valid_dataloader().reset()
        print(
            " | ".join(
                (
                    f"Epoch {self.current_epoch:02d}",
                    f"Validation loss: {self.metrics['loss'].compute().item():.3f}",
                    f"Validation accuracy: {self.metrics['validation_accuracy'].compute().item():.3f}",
                )
            )
        )

    def iterate(self):
        self.on_validation_epoch_start()
        for batch_idx, batch in enumerate(self.data_module.valid_dataloader()):
            batch = self.on_validation_batch_start(batch, batch_idx)
            loss = self.validation_step(batch, batch_idx)
            self.on_validation_batch_end(loss)
        self.on_validation_epoch_end()
