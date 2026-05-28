from pathlib import Path

import pandas as pd
import torch


from modules.data_loading.audio_data_loader import LazyAudioDataset, AudioDataLoaderModule


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
        pretrained_model: str,
    ) -> None:
        """

        Args:
            df (pd.DataFrame): DataFrame with audio metadata (file name, start_sec, end_sec).
            labels (pd.Series): Target labels for each sample.
            audio_dir (Path): Directory containing audio segment files.
            pretrained_model (str): Hugging Face model identifier for the feature extractor.
        """
        self._labels = labels.reset_index(drop=True)
        self._audio_dir = Path(audio_dir)

        loader_module = AudioDataLoaderModule(
            filepath=self._audio_dir,
            sample_rate=self.TARGET_SR,
        )
        self._lazy_dataset = loader_module.run(dataframe=df, verbose=True)["dataset"]

        self._feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)

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

        # Extract features using ASTFeatureExtractor
        try:
            # Squeeze to get a 1D vector y
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)

            waveform_np = waveform.numpy()

            #  yields a dict containing 'input_values'
            encoded_features = self._feature_extractor(waveform_np, sampling_rate=self.TARGET_SR, return_tensors="pt")
            features = encoded_features["input_values"].squeeze(0)

        except Exception as e:
            logger.warning(f"AST feature extraction failed at index {idx}: {e}")
            # Fallback tensor for AST
            features = torch.zeros((1024, 128))

        return features, label
