from pathlib import Path
from typing import Sequence

import torch
import pandas as pd

from modules.feature_extraction.mfcc_extraction import MFCCExtractorModule
from torch_datasets.datasets_base import BeeAudioDataset


def pad_collate_fn(batch):
    """Collate variable-length MFCC sequences by zero-padding to the batch maximum.

    Also returns the true (pre-padding) length of each sequence so the model can pool
    over real timesteps only and ignore the zero padding.
    """
    mels, tabulars, labels = zip(*[(b[0], b[1], b[2]) for b in batch])
    lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    max_len = max(m.shape[0] for m in mels)
    n_feat = mels[0].shape[1]
    padded = torch.zeros(len(mels), max_len, n_feat)
    for i, m in enumerate(mels):
        padded[i, : m.shape[0]] = m
    return padded, torch.stack(tabulars), torch.stack(labels), lengths


class LSTMBeeAudioDataset(BeeAudioDataset):
    def __init__(
        self,
        features: pd.DataFrame,
        audio_dir: Path,
        n_mfcc: int,
        columns_to_drop: list,
        labels: pd.Series | None = None,
    ) -> None:
        """
        Initializes the LSTMBeeAudioDataset.
        
        Args:
            features (pd.DataFrame): DataFrame containing the features, including audio metadata and tabular data.
            audio_dir (Path): Directory containing the audio files.
            n_mfcc (int): Number of MFCC features to extract.
            columns_to_drop (list): List of column names to drop from the tabular features.
            labels (pd.Series, optional): Series containing the target labels. Defaults to None.
        Returns:
            None
        """
        super().__init__(features=features, labels=labels, audio_dir=audio_dir)

        self._n_mfcc = n_mfcc

        self._feature_extractor = MFCCExtractorModule(
            n_mfcc=self._n_mfcc, sample_rate=self.TARGET_SR
        )

        present = [c for c in columns_to_drop if c in self._features.columns]
        self._tabular_features = self._features.drop(present, axis=1).select_dtypes(
            include="number"
        )

    def _get_features(
        self,
        idx: int, waveform: torch.Tensor) -> Sequence:
        sequential_features = self._feature_extractor.run(waveform, verbose=False)
        sequential_features = sequential_features["mfcc"].squeeze(0)
        sequential_features = sequential_features.transpose(-1, -2).contiguous()

        tabular_features = torch.tensor(
            self._tabular_features.iloc[idx].values, dtype=torch.float32
        )

        return [sequential_features, tabular_features]
