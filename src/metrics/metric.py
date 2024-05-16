from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import mlx.core as mx


class Metric(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._allowed_reduce_fx = [mx.mean, mx.sum, mx.concatenate]
        self.reduce_fx: Callable = mx.mean
        self._defaults: dict[str, Any] = {}

    def add_state(self, name: str, default: mx.array, reduce_fx: Callable) -> None:
        if hasattr(self, name):
            raise ValueError(f"State {name} already defined")
        if reduce_fx.__name__ not in [m.__name__ for m in self._allowed_reduce_fx]:
            raise ValueError(f"Reduction function {reduce_fx.__name__} not supported.")
        self.reduce_fx = reduce_fx
        self._defaults[name] = default
        setattr(self, name, default)

    def reset(self):
        for state_variable_name, default_value in self._defaults.items():
            setattr(self, state_variable_name, default_value)

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def compute(self):
        pass
