from abc import abstractmethod
from pathlib import Path
from typing import Sequence

import pandas as pd
from transformers import ASTFeatureExtractor
import torch

from modules.data_loading.audio_data_loader import LazyAudioDataset, AudioDataLoaderModule
from logger import logger


class BeeAudioDataset(LazyAudioDataset):
    """
    PyTorch dataset that wraps LazyAudioDataset and applies AST Feature Extraction for log-mel spectrograms.

    """

    TARGET_SR: int = 16000

    def __init__(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        audio_dir: Path,
    ) -> None:
        """

        Args:
            df (pd.DataFrame): DataFrame with audio metadata (file name, start_sec, end_sec).
            labels (pd.Series): Target labels for each sample.
            audio_dir (Path): Directory containing audio segment files.
        """
        self._labels = labels.reset_index(drop=True)
        self._audio_dir = Path(audio_dir)

        loader_module = AudioDataLoaderModule(
            filepath=self._audio_dir,
            sample_rate=self.TARGET_SR,
        )
        self._lazy_dataset = loader_module.run(dataframe=df, verbose=False)["dataset"]

    def __len__(self) -> int:
        return len(self._lazy_dataset)

    def __getitem__(self, idx: int):
        """
        Get a sample: waveform → ASTFeatureExtractor → Log-mel Spectrogram tensor + label.
        """
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

        try:
            # Squeeze to get a 1D vector y
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            
            features = self._get_features(idx, waveform)

        except Exception as e:
            logger.warning(f"AST feature extraction failed at index {idx}: {e}")
            features = [torch.zeros((1024, 128))]

        return *features, label


    @abstractmethod
    def _get_features(self, idx: int, waveform: torch.Tensor) -> Sequence:
        pass
    

