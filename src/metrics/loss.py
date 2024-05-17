from __future__ import annotations

import mlx.core as mx

from src.metrics.metric import Metric


class LossMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state("losses", default=mx.array([]), reduce_fx=mx.mean)

    def __call__(self, loss: mx.array) -> None:
        self.update(loss)

    def update(self, loss: mx.array):
        self.losses = mx.concatenate([self.losses, mx.expand_dims(loss, 0)])

    def compute(self) -> mx.array:
        return self.reduce_fx(self.losses)
