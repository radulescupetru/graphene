from __future__ import annotations

from collections.abc import Callable
from functools import partial

import mlx.core as mx
import numpy as np
from mlx import nn

from src.core.datamodule import DataModule
from src.core.trainmodule import TrainModule
from src.loops.validation_loop import ValidationLoop
from src.metrics.loss import LossMetric


class Trainer:
    def __init__(
        self,
        train_module: TrainModule,
        data_module: DataModule,
        max_epochs: int,
        run_validation_every_n_epochs: int = 1,
        run_sanity_validation: bool = True,
        **kwargs,
    ) -> None:
        self.train_module = train_module
        self.data_module = data_module
        self.max_epochs = max_epochs
        self.run_validation_every_n_epochs = run_validation_every_n_epochs
        self.run_sanity_validation = run_sanity_validation

        # Define system metrics
        self.metrics = {"loss": LossMetric()}
        # Loops
        self._validation_loop = ValidationLoop(train_module, data_module, self.metrics)

    @property
    def current_epoch(self) -> int:
        if not hasattr(self, "_current_epoch"):
            return 0
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, current_epoch) -> None:
        self._current_epoch = current_epoch
        # Set the current epoch in each loop
        for loop in [self._validation_loop]:
            loop.current_epoch = current_epoch

    def fit(self):
        # Configuration
        optimizer = self.train_module.configure_optimizers()
        model = self.train_module.model
        self.data_module.setup()
        self.train_module.setup()
        # Loops
        for epoch_number in range(self.max_epochs):
            # Set the current epoch number
            self.current_epoch = epoch_number
            if epoch_number % self.run_validation_every_n_epochs == 0 and (
                epoch_number != 0 or self.run_sanity_validation
            ):
                # Run eval loop
                self._validation_loop.iterate()
            # Run training loop
            self.train_loop(model, self.data_module.train_dataloader(), optimizer)
            self.data_module.train_dataloader().reset()

    def train_loop(self, model: nn.Module, train_dataloader, optimizer):
        """Training loop.

        Args:
            model: Model to train.
            train_dataloader: Dataloader to train on.
            optimizer: Optimizer to use.
        """
        # Set the model in training mode
        model.train(True)
        # Define the state
        state = [model.state, optimizer.state]
        # Set the user defined training step callable
        training_step_fun: Callable = self.train_module.training_step
        accs = []

        @partial(mx.compile, inputs=state, outputs=state)
        def step(batch: dict[str, np.array], batch_idx: int) -> tuple[mx.array, mx.array]:
            """Perform a training step.

            Args:
                batch (np.array): Batch of data
                batch_idx (int): Batch index

            Returns:
                tuple[mx.array, mx.array]: Tuple with loss and grads
            """
            # Modify the training step function to return grads
            training_step_fun_with_grad = nn.value_and_grad(model, training_step_fun)
            # Call the training step from the trainmodule
            (loss, acc), grads = training_step_fun_with_grad(batch, batch_idx)
            # Return the loss and grads
            return (loss, acc), grads

        for batch_index, batch in enumerate(train_dataloader):
            # Cast the numpy arrays in each batch to mx arrays
            batch = {k: mx.array(v) for k, v in batch.items()}
            (loss, acc), grads = step(batch, batch_index)
            accs.append(acc)
            optimizer.update(model, grads)
            if batch_index % 10 == 0:
                print(
                    " | ".join(
                        (f"Epoch {self.current_epoch:02d} [{batch_index:03d}]", f"Train loss: {loss.item():.3f}")
                    )
                )
            mx.eval(state)
        print(f"Epoch {self.current_epoch:02d} | Accuracy: {mx.mean(mx.array(accs)).item():.3f}")
