from abc import abstractmethod
from typing import Any

from pipeline import ModelPipelineStep

class TrainerBase(ModelPipelineStep):
    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val) -> Any:
        pass

class InferencerBase(ModelPipelineStep):
    def __init__(self) -> None:
        super().__init__()

        # Keep track of indices in case of filtering non existent x-test
        self.last_kept_index = None
        
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
