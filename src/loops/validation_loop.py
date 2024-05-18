from __future__ import annotations

import mlx.core as mx
import numpy as np

from src.core.datamodule import DataModule
from src.core.trainmodule import TrainModule
from src.loops.loop import Loop
from src.metrics import Accumulation
from src.metrics.metric import Metric


class ValidationLoop(Loop):
    def __init__(self, trainer, train_module: TrainModule, data_module: DataModule) -> None:
        super().__init__(trainer, train_module, data_module)
        self.metrics: dict[str, Metric] = {"loss": Accumulation()}

    def on_validation_epoch_start(self, *args, **kwargs):
        """Validation hook which triggers at the beginning of a validation epoch.

        - Sets the model to validation mode.
        - Calls user-defined methods and resets metrics.
        """
        self.model.train(False)
        self._call_user_method("on_validation_epoch_start", args, kwargs)
        self._reset_metrics()

    def on_validation_batch_start(self, batch: np.array, batch_idx: int) -> mx.array:
        """Validation hook which triggers at the start of a validation batch.

        - Converts the batch to mx.array format.
        - Calls user-defined methods.

        Args:
            batch : A batch of data.
            batch_idx: Index of the batch.

        Returns:
            batch: Batch formatted as mx.array with additional user-defined transformations.
        """
        batch = self._call_user_method("on_validation_batch_start", batch, batch_idx)
        batch = {k: mx.array(v) for k, v in batch.items()}
        return batch

    def on_validation_batch_end(self, loss):
        """Validation hook which triggers at the end of a validation batch.

        - Calls user-defined methods.
        - Updates the loss metric.
        """
        self._call_user_method("on_validation_batch_end", loss)
        self.metrics["loss"].update(loss)

    def validation_step(self, batch, batch_idx) -> tuple[mx.array, mx.array]:
        """Executes a validation step, computing loss.

        Args:
            batch: A batch of data.
            batch_idx: Index of the batch.

        Returns:
            Tuple containing the loss.
        """
        assert hasattr(self.train_module, "validation_step"), "The train module has no validation_step defined"
        loss = self.train_module.validation_step(batch, batch_idx)
        return loss

    def on_validation_epoch_end(self, *args, **kwargs):
        """Validation hook which triggers at the end of a validation epoch.

        - Calls user-defined methods.
        - Resets the data loader.
        - Logs the validation metrics.
        """
        self._call_user_method("on_validation_epoch_end", args, kwargs)
        self.data_module.valid_dataloader().reset()
        self._log_epoch_metrics()

    def iterate(self):
        """Main loop iteration for validation.

        - Triggers events and hooks at the start and end of each epoch and batch.
        - Executes the validation step for each batch.
        """
        epoch_events = {
            "start_event": "on_validation_epoch_start",
            "end_event": "on_validation_epoch_end",
            "start_method": self.on_validation_epoch_start,
            "end_method": self.on_validation_epoch_end,
        }

        batch_events = {
            "start_event": "on_validation_batch_start",
            "end_event": "on_validation_batch_end",
            "step_event": "validation_step",
            "start_method": self.on_validation_batch_start,
            "end_method": self.on_validation_batch_end,
        }

        self._execute_with_events(
            epoch_events=epoch_events,
            batch_events=batch_events,
            batch_iterator=self.data_module.valid_dataloader(),
            step_method=self.validation_step,
        )

    def _log_epoch_metrics(self):
        """Logs the metrics at the end of an epoch."""
        print(
            " | ".join(
                (
                    f"Epoch {self.current_epoch:02d}",
                    f"Validation loss: {self.metrics['loss'].compute().item():.3f}",
                    f"Validation accuracy: {self.metrics['validation_accuracy'].compute():.3f}",
                )
            )
        )
