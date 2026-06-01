from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from logger import console, logger
from modules.lstm.lstm_model import FusionLSTMModel
from modules.model_bases import InferencerBase
from torch_datasets.lstm_dataset import LSTMBeeAudioDataset, pad_collate_fn


class FusionLSTMInferenceModule(InferencerBase):
    name = "LstmInference"
    inputs = {"model", "x_test"}
    outputs = {"y_pred", "y_pred_proba"}

    def __init__(
        self,
        drop_column: list[str] = ["file name"],
        n_mfcc: int = 40,
        audio_dir: str = "src/data/sound_files/sound_files",
        audio_path_col: str = "file name",
        batch_size: int = 32,
        num_workers: int = 0,
        threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        """
        Initialise the lstm inference step.

        Args:
            audio_dir (str)         : Directory containing audio files.
            audio_path_col (str)    : Column in x_test holding the audio file path.
            batch_size (int)        : DataLoader batch size.
            num_workers (int)       : DataLoader worker processes.
            device (str | None)     : 'cpu', 'cuda', 'mps', or None (auto-detect).
        Returns:
            None
        """
        super().__init__()

        self._audio_dir = Path(audio_dir)
        self._audio_col = audio_path_col
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._n_mfcc = n_mfcc
        self._columns_to_drop = drop_column
        self._threshold = threshold

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

    def _build_loader(
        self,
        x_test: pd.DataFrame,
    ) -> DataLoader:
        """
        Build a DataLoader for the test set.
        
        Args:
            x_test (pd.DataFrame): Test features with audio metadata and tabular data.
        Returns:
            DataLoader: A DataLoader yielding batches of (mel, tabular, labels, lengths).
        """
        dataset = LSTMBeeAudioDataset(
            columns_to_drop=self._columns_to_drop,
            features=x_test,
            audio_dir=self._audio_dir,
            n_mfcc=self._n_mfcc,
        )
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=(self._device.type == "cuda"),
            collate_fn=pad_collate_fn,
        )

    def run(
        self,
        model: FusionLSTMModel,
        x_test: pd.DataFrame,
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        """
        Run batch inference and return predictions + probabilities.

        Args:
            model   (FusionLSTMModel): The fusion lstm model.
            x_test  (pd.DataFrame)    : Test features with audio metadata and tabular data.
            verbose (bool)            : Verbose logging. Defaults to True.
        Returns:
            dict containing:
                y_pred       - predicted class indices
                y_pred_proba - probability of class 1
        """

        if verbose:
            console.section("Evaluating LSTM Fusion model")
            logger.info(f"Device     : {self._device}")
            logger.info(f"Test shape : {x_test.shape}")

        model = model.to(self._device)
        model.eval()

        loader = self._build_loader(x_test)
        self.last_kept_index = loader.dataset.kept_index

        all_preds, all_proba = [], []
        with torch.no_grad():
            for batch_mel, batch_tabular, _, _ in loader:
                batch_mel = batch_mel.to(self._device)
                batch_tabular = batch_tabular.to(self._device)

                logits = model(batch_mel, batch_tabular)
                probs = torch.sigmoid(logits.squeeze(1))
                preds = (probs > self._threshold).long()

                all_preds.append(preds.cpu().numpy())
                all_proba.append(probs.cpu().numpy())

        if not all_preds:
            logger.warning(
                "Inference produced no predictions. This may indicate an issue" \
                "with the test data or DataLoader configuration."
            )
            return {"y_pred": np.array([]), "y_pred_proba": np.array([])}

        y_pred = np.concatenate(all_preds)
        y_pred_proba = np.concatenate(all_proba)

        if verbose:
            logger.info(f"Finished inference. Prediction shape: {y_pred.shape}")

        return {
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

    def inference(
        self,
        model: FusionLSTMModel,
        x_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return predicted class labels for x_test.

        Args:
            model  (FusionLSTMModel): The LSTM + MLP model.
            x_test (pd.DataFrame): Test features with audio metadata.
        Returns:
            np.ndarray: Predicted class indices
        """
        return self.run(model=model, x_test=x_test, verbose=False)["y_pred"]

    def inference_proba(
        self,
        model: FusionLSTMModel,
        x_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return predicted probabilities (positive class) for x_test.

        Args:
            model  (FusionLSTMModel): The LSTM + MLP model.
            x_test (pd.DataFrame): Test features with audio metadata.
        Returns:
            np.ndarray: Probability of class 1, shape (N,).
        """
        return self.run(model=model, x_test=x_test, verbose=False)["y_pred_proba"]
