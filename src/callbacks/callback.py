from __future__ import annotations


class Callback:
    def on_fit_start(self, trainer):
        pass

    def on_fit_end(self, trainer):
        pass

    def on_train_epoch_start(self, trainer):
        pass

    def on_train_epoch_end(self, trainer):
        pass

    def on_train_batch_start(self, trainer):
        pass

    def training_step(self, trainer):
        pass

    def on_train_batch_end(self, trainer):
        pass

    def on_validation_epoch_start(self, trainer):
        pass

    def on_validation_step_start(self, trainer):
        pass

    def on_validation_step_end(self, trainer):
        pass

    def on_validation_epoch_end(self, trainer):
        pass
