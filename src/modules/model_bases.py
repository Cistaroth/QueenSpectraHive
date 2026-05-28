from abc import abstractmethod
from typing import Any

from pipeline import ModelPipelineStep

class TrainerBase(ModelPipelineStep):
    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val) -> Any:
        pass

class InferencerBase(ModelPipelineStep):
    @abstractmethod
    def inference(self, model, x_test) -> Any:
        pass

    @abstractmethod
    def inference_proba(self, model, x_test) -> Any:
        pass


class TransformBase(ModelPipelineStep):
    @abstractmethod
    def fit_transform(self, x) -> Any:
        pass

    @abstractmethod
    def transform(self, x) -> Any:
        pass
