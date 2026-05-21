from abc import abstractmethod
from typing import Any

from pipeline import ModelPipelineStep


class TrainerBase(ModelPipelineStep):
    @abstractmethod
    def train(self, x_train, y_train) -> Any:
        pass


class InferencerBase(ModelPipelineStep):
    @abstractmethod
    def inference(self, model, x_test_scaled) -> Any:
        pass

    @abstractmethod
    def inference_proba(self, model, x_test_scaled) -> Any:
        pass


class ScalerBase(ModelPipelineStep):
    def scale(self, x_train, x_val) -> Any:
        return self.fit_transform(x=x_train), self.transform(x=x_val)

    @abstractmethod
    def fit_transform(self, x) -> Any:
        pass

    @abstractmethod
    def transform(self, x) -> Any:
        pass
