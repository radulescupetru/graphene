from __future__ import annotations

from abc import ABC, abstractmethod

from mlx import nn

from src.core.datamodule import DataModule
from src.core.trainmodule import TrainModule


class Loop(ABC):
    """Base class for loops."""

    def __init__(self, train_module: TrainModule, data_module: DataModule) -> None:
        super().__init__()
        self.train_module = train_module
        self.data_module = data_module

    def setup(self):
        """Method which makes sure both the train module and the data modules are properly
        setup."""
        self.data_module.setup()
        self.train_module.setup()

    @property
    def model(self) -> nn.Module:
        return self.train_module.model

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, current_epoch) -> None:
        self._current_epoch = current_epoch

    @abstractmethod
    def iterate(self):
        raise NotImplementedError
