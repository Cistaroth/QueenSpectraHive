import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logger import console, logger
from modules.data_loading.audio_data_loader import AudioDataLoaderModule, LazyAudioDataset
from modules.feature_extraction.mfcc_extraction import MFCCExtractorModule
from modules.model_bases import TrainerBase


class CompositeModel(
    nn.Module,
):
    def __init__(self, lstm_layers: tuple, cur_input_size: int, ff_hidden_size, layers):
        super().__init__()

        seq_input_size, lstm_hidden_size, lstm_num_layers, lstm_dropout = lstm_layers

        # LSTM Component
        self.lstm = nn.LSTM(seq_input_size, lstm_hidden_size, lstm_num_layers, dropout=lstm_dropout, batch_first=True)

        # Tabular component
        self.embeddings_model = nn.Sequential(*layers)

        # Feed-forward Component
        self.ff = nn.Sequential(nn.Linear(lstm_hidden_size + cur_input_size, ff_hidden_size), nn.ReLU(), nn.Linear(ff_hidden_size, 1))

    def forward(self, historical_data: torch.Tensor, current_data: torch.Tensor):
        lstm_output, _ = self.lstm(historical_data)

        tabular_embedding_output = self.embeddings_model(current_data)

        # Select the last LSTM output (final time step)
        lstm_last_output = lstm_output[:, -1, :]

        # Concatenate the LSTM output with current features
        combined_features = torch.cat([lstm_last_output, tabular_embedding_output], dim=1)

        # Pass through the feed-forward component for final prediction
        result = self.ff(combined_features)
        return result


class CompositeModelModule(TrainerBase):
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
        num_workers: int,
        seed: int,
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

    def run(self, x_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True) -> dict[str, CompositeModel]:

        if verbose:
            console.section(title="Generating Composite model")
            logger.info(f"Processing on: {x_train.shape[-1] - len(self._drop_column)} columns")

        dataset = BeeAudioDataset(df=x_train, labels=y_train, audio_dir=self._audio_dir, n_mfcc=self._n_mfcc, columns_to_drop=self._drop_column)

        loader = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, pin_memory=(self._device.type == "cuda")
        )

        model = CompositeModel(
            lstm_layers=self._lstm_layers, cur_input_size=32, ff_hidden_size=self._classification_hidden_size, layers=self._layers
        )  # Seq_input_size, lstm_hidden_size, lstm_num_layers, lstm_dropout

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

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame) -> CompositeModel:
        """
        Obtain the embeddings for the tabular data.
        """
        return self.run(x_train=x_train, y_train=y_train, verbose=False)["model"]


class BeeAudioDataset(LazyAudioDataset):
    """

    #! CAN DEFINITELY BE MERGED WITH THE BEEAUDIODATASET USED BY EMIL IN THE TRANSFORMER, I ADDED |NONE SO THAT IT CAN ALSO BE USED IN INFERENCE
    PyTorch dataset that wraps LazyAudioDataset and applies AST Feature Extraction for log-mel spectrograms.

    """

    TARGET_SR: int = 16000

    def __init__(self, df: pd.DataFrame, labels: pd.Series|None, audio_dir: Path, n_mfcc: int, columns_to_drop: list) -> None:
        """

        Args:
            df (pd.DataFrame): DataFrame with audio metadata (file name, start_sec, end_sec).
            labels (pd.Series): Target labels for each sample.
            audio_dir (Path): Directory containing audio segment files.
            pretrained_model (str): Hugging Face model identifier for the feature extractor.
        """
        if labels is None:
            self._labels = None
        else:
            self._labels = labels.reset_index(drop=True)
        self._audio_dir = Path(audio_dir)
        self._n_mfcc = n_mfcc

        loader_module = AudioDataLoaderModule(
            filepath=self._audio_dir,
            sample_rate=self.TARGET_SR,
        )
        self._lazy_dataset = loader_module.run(dataframe=df, verbose=False)["dataset"]

        self._feature_extractor = MFCCExtractorModule(n_mfcc=self._n_mfcc)

        self._tabular_features = df.drop(columns_to_drop, axis=1)

    def __len__(self) -> int:
        return len(self._lazy_dataset)

    def __getitem__(self, idx: int):
        """
        Get a sample: waveform → ASTFeatureExtractor → Log-mel Spectrogram tensor + label.
        """
        if self._labels is None:
            label = None
        else:
            label = int(self._labels.iloc[idx])

        try:
            # Get waveform from LazyAudioDataset via AudioDataLoaderModule (shape: [channels, time])
            waveform = self._lazy_dataset[idx]
            logger.debug(f"Loaded waveform at index {idx} with shape {waveform.shape} and dtype {waveform.dtype}")
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)
        except Exception as e:
            logger.warning(f"Failed to load waveform at index {idx}: {e}. Using silence.")
            waveform = torch.zeros((1, self.TARGET_SR * 15))

        # Extract features using ASTFeatureExtractor
        try:
            # Squeeze to get a 1D vector y
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)

            #  yields a dict containing 'input_values'
            encoded_features = self._feature_extractor.run(waveform, sample_rate=self.TARGET_SR, verbose=False)
            features = encoded_features["mfcc"].squeeze(0)
            features = features.transpose(-1, -2).contiguous()

        except Exception as e:
            logger.warning(f"AST feature extraction failed at index {idx}: {e}")
            # Fallback tensor for AST
            features = torch.zeros((1024, 128))

        return features, torch.tensor(self._tabular_features.iloc[idx, :].values, dtype=torch.float32), label
