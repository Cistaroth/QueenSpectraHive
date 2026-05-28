import pandas as pd
import torch
import torch.nn as nn

from logger import console, logger
from modules.model_bases import TrainerBase


class TabularNNEmbeddingsModule(TrainerBase):
    name = "TabularNNEmbeddings"
    inputs = {"x_train"}
    outputs = {"tabular_embedding"}

    def __init__(self, drop_column: str, layers: tuple) -> None:
        """
        Initialize the Neural Network to combine tabular data with the LSTM trainer class

        Args:
            *args: Unpacked list of PyTorch layers
        """
        super().__init__()
        self._drop_column = drop_column
        self._layers = layers

    def run(self, x_train: pd.DataFrame, verbose: bool = True) -> dict[str, torch.Tensor]:

        if verbose:
            console.section(title="Generating Sequential NN Embeddings")
            logger.info(f"Processing on: {x_train.shape[-1] - 1} columns")

        x_train = x_train.drop(self._drop_column, axis=1)
        model = nn.Sequential(*self._layers)

        x_tensor = torch.tensor(x_train.values, dtype=torch.float32)

        tabular_embedding = model(x_tensor)

        if verbose:
            logger.info("Finished generating tabular embeddings.")


        return {"tabular_embedding": tabular_embedding}

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> torch.Tensor:
        """
        Obtain the embeddings for the tabular data.
        """
        return self.run(x_train=x_train, y_train=y_train, verbose=False)["tabular_embedding"]
