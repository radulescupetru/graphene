from __future__ import annotations

from abc import ABC, abstractmethod


class DataModule(ABC):
    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def train_dataloader(self):
        pass

    @abstractmethod
    def valid_dataloader(self):
        pass

    @abstractmethod
    def test_dataloader(self):
        pass
