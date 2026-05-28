from pathlib import Path
from typing import Sequence

import torch
import pandas as pd

from transformers import ASTFeatureExtractor
from torch_datasets.datasets_base import BeeAudioDataset

class TransformerBeeAudioDataset(BeeAudioDataset):
    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        audio_dir: Path,
        pretrained_model: str,
    ):
        super().__init__(features=features, labels=labels, audio_dir=audio_dir)

        self._feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

    def _get_features(self, idx: int, waveform: torch.Tensor) -> Sequence:
        waveform_np = waveform.numpy()

        features = self._feature_extractor(
                waveform_np, 
                sampling_rate=self.TARGET_SR, 
                return_tensors="pt"
            )
        
        features = features["input_values"].squeeze(0)

        return [features]
    


