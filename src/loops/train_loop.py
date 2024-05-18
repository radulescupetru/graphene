from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from src.core.datamodule import DataModule
from src.core.trainmodule import TrainModule
from src.loops.loop import Loop
from src.metrics import Accumulation
from src.metrics.metric import Metric


class TrainLoop(Loop):
    def __init__(self, trainer, train_module: TrainModule, data_module: DataModule) -> None:
        super().__init__(trainer, train_module, data_module)
        self.metrics: dict[str, Metric] = {"loss": Accumulation()}

    def on_train_epoch_start(self, *args, **kwargs):
        """Training hook which triggers at the beginning of a training epoch.

        - Sets the model to training mode.
        - Initializes the state.
        - Calls user-defined methods and resets metrics.
        """
        self.model.train(True)
        self.state = [self.model.state, self.trainer.optimizer.state]
        self._call_user_method("on_train_epoch_start", args, kwargs)
        self._reset_metrics()

    def on_train_batch_start(self, batch: np.array, batch_idx: int) -> mx.array:
        """Training hook which triggers at the start of a training batch.

        - Converts the batch to mx.array format.
        - Calls user-defined methods.

        Args:
            batch : A batch of data.
            batch_idx: Index of the batch.

        Returns:
            batch: Batch formatted as mx.array with additional user-defined transformations.
        """
        batch = self._call_user_method("on_train_batch_start", batch, batch_idx)
        batch = {k: mx.array(v) for k, v in batch.items()}
        return batch

    def on_train_batch_end(self, loss):
        """Training hook which triggers at the end of a training batch.

        - Calls user-defined methods.
        - Evaluates the model state and updates metrics.
        """
        self._call_user_method("on_train_batch_end", loss)
        mx.eval(self.state)
        self.metrics["loss"].update(loss)

    def training_step(self, batch, batch_idx) -> tuple[mx.array, mx.array]:
        """Executes a training step, computing loss and gradients.

        Args:
            batch: A batch of data.
            batch_idx: Index of the batch.

        Returns:
            Tuple containing the loss and gradients.
        """
        assert hasattr(self.train_module, "training_step"), "The train module has no training_step defined"
        loss, grads = nn.value_and_grad(self.model, self.train_module.training_step)(batch, batch_idx)
        self.trainer.optimizer.update(self.model, grads)
        return loss

    def on_train_epoch_end(self, *args, **kwargs):
        """Training hook which triggers at the end of a training epoch.

        - Calls user-defined methods.
        - Resets the data loader.
        - Logs the training metrics.
        """
        self._call_user_method("on_train_epoch_end", args, kwargs)
        self.data_module.train_dataloader().reset()
        self._log_epoch_metrics()

    def iterate(self):
        """Main loop iteration for training.

        - Triggers events and hooks at the start and end of each epoch and batch.
        - Executes the training step for each batch.
        """
        epoch_events = {
            "start_event": "on_train_epoch_start",
            "end_event": "on_train_epoch_end",
            "start_method": self.on_train_epoch_start,
            "end_method": self.on_train_epoch_end,
        }

        batch_events = {
            "start_event": "on_train_batch_start",
            "end_event": "on_train_batch_end",
            "step_event": "training_step",
            "start_method": self.on_train_batch_start,
            "end_method": self.on_train_batch_end,
        }

        self._execute_with_events(
            epoch_events=epoch_events,
            batch_events=batch_events,
            batch_iterator=self.data_module.train_dataloader(),
            step_method=self.training_step,
        )

    def _log_epoch_metrics(self):
        """Logs the metrics at the end of an epoch."""
        print(
            " | ".join(
                (
                    f"Epoch {self.current_epoch:02d}",
                    f"Training loss: {self.metrics['loss'].compute().item():.3f}",
                    f"Training accuracy: {self.metrics['train_accuracy'].compute():.3f}",
                )
            )
        )
