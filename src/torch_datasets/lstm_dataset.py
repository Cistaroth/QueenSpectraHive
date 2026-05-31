from pathlib import Path
from typing import Sequence

import torch
import pandas as pd

from modules.feature_extraction.mfcc_extraction import MFCCExtractorModule
from torch_datasets.datasets_base import BeeAudioDataset


class LSTMBeeAudioDataset(BeeAudioDataset):
    def __init__(
        self,
        features: pd.DataFrame,
        audio_dir: Path,
        n_mfcc: int,
        columns_to_drop: list,
        labels: pd.Series | None = None,
    ):
        super().__init__(features=features, labels=labels, audio_dir=audio_dir)

        self._n_mfcc = n_mfcc

        self._feature_extractor = MFCCExtractorModule(n_mfcc=self._n_mfcc, sample_rate=self.TARGET_SR)

        present = [c for c in columns_to_drop if c in self._features.columns]
        self._tabular_features = self._features.drop(present, axis=1).select_dtypes(include="number")

    def _get_features(self, idx: int, waveform: torch.Tensor) -> Sequence:
        sequential_features = self._feature_extractor.run(waveform, verbose=False)
        sequential_features = sequential_features["mfcc"].squeeze(0)
        sequential_features = sequential_features.transpose(-1, -2).contiguous()

        tabular_features = torch.tensor(self._tabular_features.iloc[idx].values, dtype=torch.float32)
        
        return [sequential_features, tabular_features]
    


