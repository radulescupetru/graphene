from __future__ import annotations

import wandb

from graphene.callbacks import Callback
from graphene.loops.loop import LoopType


class WandbCallback(Callback):
    def __init__(self, project_name: str = "graphene", **kwargs) -> None:
        self.project_name = project_name
        wandb.init(project=self.project_name, **kwargs)

    def on_fit_start(self, trainer) -> None:
        wandb.config.update(
            {
                "max_epochs": trainer.max_epochs,
                "run_validation_every_n_epochs": trainer.run_validation_every_n_epochs,
                "run_sanity_validation": trainer.run_sanity_validation,
            }
        )

    def on_train_epoch_end(self, trainer):
        metrics = trainer.loops[LoopType.TRAINING].metrics
        current_step = trainer.optimizer.step.item()
        self.on_epoch_end(metrics, trainer.active_loop.name, current_step)
        wandb.log({"epoch": trainer.current_epoch}, step=current_step)

    def on_validation_epoch_end(self, trainer):
        metrics = trainer.loops[LoopType.VALIDATION].metrics
        current_step = trainer.optimizer.step.item()
        self.on_epoch_end(metrics, trainer.active_loop.name, current_step)

    def on_epoch_end(self, metrics, active_loop_name: str, current_step: int) -> None:
        for name, metric in metrics.items():
            wandb.log({f"{active_loop_name}/{name}": metric.compute()}, step=current_step)

    def on_train_batch_end(self, trainer, batch_idx: int) -> None:
        metrics = trainer.loops[LoopType.TRAINING].metrics
        current_step = trainer.optimizer.step.item()

        wandb.log({"lr": trainer.optimizer.learning_rate.item()}, step=current_step)
        self.on_batch_end(metrics, trainer.active_loop.name, current_step)

    def on_batch_end(self, metrics, active_loop_name, batch_idx):
        for name, metric in metrics.items():
            wandb.log({f"{active_loop_name}/{name}_batch": metric.compute()}, step=batch_idx)

    def on_fit_end(self, trainer):
        wandb.finish()
