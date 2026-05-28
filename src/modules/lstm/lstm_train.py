import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from modules.model_bases import TrainerBase
from logger import console, logger
from torch_datasets.lstm_dataset import LSTMBeeAudioDataset
from modules.lstm.lstm_model import FusionLSTMModel

class FusionLSTMTrainModule(TrainerBase):
    name = "CompositeModule"
    inputs = {"x_train", "y_train"}
    outputs = {"model"}

    def __init__(
        self,
        drop_column: list[str],
        layers: tuple,
        lstm_layers: tuple,
        classification_hidden_size: int,
        audio_dir: Path,
        n_mfcc: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        num_workers: int = 0,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        """
        Initialize the Neural Network to combine tabular data with the LSTM trainer class

        Args:
            *args: Unpacked list of PyTorch layers
        """
        super().__init__()
        self._drop_column = drop_column
        self._layers = layers
        self._lstm_layers = lstm_layers
        self._classification_hidden_size = classification_hidden_size
        self._audio_dir = audio_dir
        self._n_mfcc = n_mfcc

        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._num_workers = num_workers
        self._seed = seed

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

    def _set_seed(self) -> None:
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(self._seed)
    
    def run(self, x_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True) -> dict[str, FusionLSTMModel]:

        if verbose:
            console.section(title="Generating Composite model")
            logger.info(f"Processing on: {x_train.shape[-1] - len(self._drop_column)} columns")

        dataset = LSTMBeeAudioDataset(df=x_train, labels=y_train, audio_dir=self._audio_dir, n_mfcc=self._n_mfcc, columns_to_drop=self._drop_column)

        loader = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, pin_memory=(self._device.type == "cuda")
        )

        model = FusionLSTMModel(
            lstm_layers=self._lstm_layers, cur_input_size=32, ff_hidden_size=self._classification_hidden_size, layers=self._layers
        )

        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self._epochs)

        for epoch in range(1, self._epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            if verbose:
                logger.info(f"Epoch {epoch}/{self._epochs} - Training...")

            for batch_features, batch_tabular_features, batch_labels in loader:
                batch_features = batch_features.to(self._device)
                batch_labels = batch_labels.to(self._device)
                batch_tabular_features = batch_tabular_features.to(self._device)

                optimiser.zero_grad()
                logits = model(batch_features, batch_tabular_features)
                loss = criterion(logits.squeeze(1), batch_labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

                running_loss += loss.item() * len(batch_labels)
                preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

            scheduler.step()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            if verbose:
                logger.info(f"Epoch completed [{epoch:>3}/{self._epochs}]  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

        if verbose:
            logger.info("Finished creating composite model.")

        return {"model": model}

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> FusionLSTMModel:
        """
        Obtain the embeddings for the tabular data.
        """
        return self.run(x_train=x_train, y_train=y_train, verbose=False)["model"]
