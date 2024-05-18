from __future__ import annotations

from contextlib import contextmanager

from mlx import nn

from src.callbacks.model_summary import ModelSummary
from src.core.datamodule import DataModule
from src.core.trainmodule import TrainModule
from src.loops.loop import LoopType
from src.loops.train_loop import TrainLoop
from src.loops.validation_loop import ValidationLoop


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
        self.train_module.trainer = self  # Set reference to the trainer inside the train module
        self.data_module = data_module
        self.max_epochs = max_epochs
        self.run_validation_every_n_epochs = run_validation_every_n_epochs
        self.run_sanity_validation = run_sanity_validation

        # Initialize model and optimizer
        self.model = train_module.model
        self.optimizer = self.train_module.configure_optimizers()

        # Initialize callbacks
        self.callbacks = [ModelSummary()]

        # Initialize loops
        self.loops = {
            LoopType.VALIDATION: ValidationLoop(self, train_module, data_module),
            LoopType.TRAINING: TrainLoop(self, train_module, data_module),
        }

    # Property for model
    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    # Property for current epoch
    @property
    def current_epoch(self) -> int:
        if not hasattr(self, "_current_epoch"):
            return 0
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, current_epoch) -> None:
        self._current_epoch = current_epoch
        # Set the current epoch in each loop
        for loop in self.loops.values():
            loop.current_epoch = current_epoch

    # Property for active loop
    @property
    def active_loop(self) -> LoopType:
        return self._active_loop

    @active_loop.setter
    def active_loop(self, active_loop: LoopType):
        self._active_loop = active_loop

    # Property for optimizer
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @contextmanager
    def loop_context(self, loop_type: LoopType):
        """Context manager to set the active loop type."""
        self.active_loop = loop_type
        try:
            yield
        finally:
            self.active_loop = LoopType.TRAINING

    def log(self, name, value):
        """Logs a value during the active loop."""
        active_loop = self.loops[self.active_loop]
        active_loop.log(name, value)

    def register_callback(self, callback):
        """Registers a new callback."""
        self.callbacks.append(callback)

    def _trigger_event(self, event):
        """Triggers a specific event for all registered callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(self)

    def fit(self):
        """Runs the fit process for the given number of epochs.

        Handles setup, sanity validation, training, and validation loops.
        """
        self._setup()
        self._trigger_event("on_fit_start")

        for epoch_number in range(self.max_epochs):
            self.current_epoch = epoch_number

            if self._should_run_sanity_validation():
                self._run_sanity_validation()

            self._run_training_loop()

            if self._should_run_validation():
                self._run_validation_loop()

    def _setup(self):
        """Configures the data and training modules."""
        self.data_module.setup()
        self.train_module.setup()

    def _should_run_sanity_validation(self):
        """Determines if sanity validation should run."""
        return self.run_sanity_validation and self.current_epoch == 0

    def _run_sanity_validation(self):
        """Runs the sanity validation loop."""
        with self.loop_context(LoopType.VALIDATION):
            self.loops[LoopType.VALIDATION].iterate()

    def _run_training_loop(self):
        """Runs the training loop for the current epoch."""
        with self.loop_context(LoopType.TRAINING):
            self.loops[LoopType.TRAINING].iterate()

    def _should_run_validation(self):
        """Determines if validation should run this epoch."""
        return self.current_epoch % self.run_validation_every_n_epochs == 0

    def _run_validation_loop(self):
        """Runs the validation loop for the current epoch."""
        with self.loop_context(LoopType.VALIDATION):
            self.loops[LoopType.VALIDATION].iterate()
