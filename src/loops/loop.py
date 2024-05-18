from __future__ import annotations

import enum
from abc import ABC, abstractmethod

from mlx import nn

from src.core.datamodule import DataModule
from src.core.trainmodule import TrainModule


class LoopType(enum.Enum):
    TRAINING = enum.auto()
    VALIDATION = enum.auto()


class Loop(ABC):
    """Base class for loops."""

    def __init__(self, trainer, train_module: TrainModule, data_module: DataModule) -> None:
        super().__init__()
        self.train_module = train_module
        self.data_module = data_module
        self.trainer = trainer

    def setup(self):
        """Method which makes sure both the train module and the data modules are properly
        setup."""
        self.data_module.setup()
        self.train_module.setup()

    def log(self, name, value):
        self.metrics[name] = value

    # Helper methods
    def _call_user_method(self, method_name, *args, **kwargs):
        """Calls a user-defined method if it exists."""
        if hasattr(self.train_module, method_name):
            return getattr(self.train_module, method_name)(*args, **kwargs)
        return args[0] if args else None

    def _reset_metrics(self):
        """Resets all metrics at the start of an epoch."""
        for metric in self.metrics.values():
            metric.reset()

    @property
    def model(self) -> nn.Module:
        return self.train_module.model

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, current_epoch) -> None:
        self._current_epoch = current_epoch

    def _execute_with_events(self, epoch_events, batch_events, batch_iterator, step_method):
        """Helper method to execute loop iterations with events.

        Args:
            epoch_events (dict): Dictionary containing start and end events and methods for the epoch.
                {
                    "start_event": str,
                    "end_event": str,
                    "start_method": callable,
                    "end_method": callable
                }
            batch_events (dict): Dictionary containing start, end, and step events and methods for the batch.
                {
                    "start_event": str,
                    "end_event": str,
                    "step_event": str,
                    "start_method": callable,
                    "end_method": callable
                }
            batch_iterator (iterable): Iterator for the batches.
            step_method (callable): Method to perform the step operation.
        """
        # Trigger start of epoch events and methods
        self.trainer._trigger_event(epoch_events["start_event"])
        epoch_events["start_method"]()

        for batch_idx, batch in enumerate(batch_iterator):
            # Trigger start of batch events and methods
            self.trainer._trigger_event(batch_events["start_event"])
            batch = batch_events["start_method"](batch, batch_idx)

            # Perform the step operation
            loss = step_method(batch, batch_idx)
            self.trainer._trigger_event(batch_events["step_event"])

            # Trigger end of batch events and methods
            batch_events["end_method"](loss)
            self.trainer._trigger_event(batch_events["end_event"])

        # Trigger end of epoch events and methods
        epoch_events["end_method"]()
        self.trainer._trigger_event(epoch_events["end_event"])

    @abstractmethod
    def iterate(self):
        raise NotImplementedError
