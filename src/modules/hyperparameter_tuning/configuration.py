from dataclasses import dataclass, field
from itertools import product
from typing import Any

from modules.model_bases import TrainerBase, InferencerBase, TransformBase
from modules.utils.passthrough_transform import PassThroughTransformModule

@dataclass(frozen=True)
class FineTuningConfiguration:
    """
    Configuration for fine-tuning a model with stratified k-fold cross-validation.
    """

    model_name: str
    model_train: type[TrainerBase]
    model_inference: type[InferencerBase]

    metric: str
    folds: int = 5

    transformer: type[TransformBase] = PassThroughTransformModule
    resampler: type | None = None
    model_hyperparameters: dict[str, list[Any]] = field(default_factory=dict)
    transformer_hyperparameters: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def model_hyperparameter_grid(self) -> list[dict[str, Any]]:
        keys = self.model_hyperparameters.keys()
        values = self.model_hyperparameters.values()
        combos = list(product(*values))
        return [dict(zip(keys, combo)) for combo in combos] if combos else [{}]

    @property
    def transformer_hyperparameter_grid(self) -> list[dict[str, Any]]:
        keys = self.transformer_hyperparameters.keys()
        values = self.transformer_hyperparameters.values()
        combos = list(product(*values))
        return [dict(zip(keys, combo)) for combo in combos] if combos else [{}]
