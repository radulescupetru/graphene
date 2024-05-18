from __future__ import annotations

from src.callbacks.callback import Callback


class ModelSummary(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_fit_start(self, trainer):
        print(trainer.model)
