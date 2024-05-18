from __future__ import annotations

import mlx.core as mx

from src.metrics.metric import Metric


class Accumulation(Metric):
    def __init__(self) -> None:
        super().__init__("accumulation")
        self.add_state("quantity", default=mx.array([]), reduce_fx=mx.mean)

    def update(self, quantity: mx.array):
        self.quantity = mx.concatenate([self.quantity, mx.expand_dims(quantity, 0)])

    def compute(self) -> mx.array:
        return self.reduce_fx(self.quantity)
