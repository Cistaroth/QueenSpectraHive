from dataclasses import dataclass
from itertools import product
from typing import Any

from modules.model_bases import TrainerBase, InferencerBase, TransformBase

@dataclass(frozen=True)
class FineTuningConfiguration:
    """
    Configuration for fine-tuning a model with stratified k-fold cross-validation.
    """

    model_name: str
    model_train: type[TrainerBase]
    model_inference: type[InferencerBase]
    transformer: type[TransformBase]
    hyperparameters: dict[str, list[Any]]

    metric: str
    folds: int = 5

    @property
    def hyperparameter_grid(self) -> list[dict[str, Any]]:
        keys = self.hyperparameters.keys()
        values = self.hyperparameters.values()

        return [dict(zip(keys, combo)) for combo in product(*values)]
