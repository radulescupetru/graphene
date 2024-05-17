from __future__ import annotations

from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any

from mlx import nn


class TrainModule(ABC):
    def __init__(self, args: SimpleNamespace) -> None:
        super().__init__()
        self.args = args

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    @property
    def model(self):
        return self._model

    @model.getter
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            raise ValueError("Model not defined, please set the model in your trainmodule.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def log(self, name, value):
        self.trainer.log(name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Same as :meth:`torch.nn.Module.forward`.

        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Your model's output
        """
        return super().forward(*args, **kwargs)

    @abstractmethod
    def training_step(self, batch: dict, batch_index: int, *args, **kwargs):
        """Here you compute and return the training loss and some additional metrics for e.g. the
        progress bar or logger.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.

        Return:
            - :class:`~mx.Array` - The loss tensor
            - ``dict`` - A dictionary which can include any keys, but must include the key ``'loss'`` in the case of
              automatic optimization.

        In this step you'd normally do the forward pass and calculate the loss for a batch.
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        r"""Choose what optimizers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        """
        pass
