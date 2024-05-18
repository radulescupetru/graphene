from __future__ import annotations

import mlx.core as mx

from src.metrics.metric import Metric


class Accuracy(Metric):
    def __init__(self):
        super().__init__("accuracy")
        self.add_state("correct", 0, reduce_fx=None)
        self.add_state("total", 0, reduce_fx=None)

    def update(self, predictions, targets):
        self.correct += (mx.argmax(predictions, axis=1) == targets).sum().item()
        self.total += targets.shape[0]

    def compute(self):
        return self.correct / self.total if self.total > 0 else 0.0
