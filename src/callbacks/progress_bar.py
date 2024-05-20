from __future__ import annotations

from tqdm import tqdm

from src.loops.loop import LoopType


class ProgressCallback:
    def __init__(self):
        self.train_progress_bar = None
        self.validation_progress_bar = None

    def on_fit_start(self, trainer):
        pass

    def on_train_epoch_start(self, trainer):
        self.train_progress_bar = tqdm(
            total=(
                len(trainer.data_module.train_dataloader())
                if hasattr(trainer.data_module.train_dataloader(), "len")
                else float("inf")
            ),
            desc=f"Training Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
            leave=False,
        )

    def on_train_batch_end(self, trainer):
        self.train_progress_bar.update(1)
        self._log_metrics(trainer.loops[LoopType.TRAINING].metrics, self.train_progress_bar, on_step=True)

    def on_train_epoch_end(self, trainer):
        self.train_progress_bar.close()
        self._log_metrics(trainer.loops[LoopType.TRAINING].metrics, None, on_epoch=True)

    def on_validation_epoch_start(self, trainer):
        self.validation_progress_bar = tqdm(
            total=(
                len(trainer.data_module.valid_dataloader())
                if hasattr(trainer.data_module.valid_dataloader(), "len")
                else float("inf")
            ),
            desc=f"Validation Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
            leave=False,
        )

    def on_validation_batch_end(self, trainer):
        self.validation_progress_bar.update(1)
        self._log_metrics(trainer.loops[LoopType.VALIDATION].metrics, self.validation_progress_bar, on_step=True)

    def on_validation_epoch_end(self, trainer):
        self.validation_progress_bar.close()
        self._log_metrics(trainer.loops[LoopType.VALIDATION].metrics, None, on_epoch=True)

    def _log_metrics(self, metrics, progress_bar, on_step=False, on_epoch=False):
        postfix_dict = {}
        for name, metric in metrics.items():
            if on_step and metric.on_step:
                value = metric.compute()
                postfix_dict[name] = value

            if on_epoch and metric.on_epoch:
                value = metric.compute()
                print(f"{name}: {value:.4f}")

        if progress_bar is not None and postfix_dict:
            progress_bar.set_postfix(postfix_dict)
