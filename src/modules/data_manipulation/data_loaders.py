from pathlib import Path
import soundfile as sf
import pandas as pd

from pipeline import ModelPipelineStep
from logger import console, logger

import torchaudio
from torch.utils.data import Dataset

import torch


class TabularDataLoaderModule(ModelPipelineStep):
    name = "TabularDataLoader"
    inputs = {}
    outputs = {"dataframe"}

    def __init__(
        self,
        filepath: Path = Path(__file__).parent.parent / "data" / "all_data_updated.csv",
    ) -> None:
        self._filepath = filepath

        super().__init__()

    def run(self, verbose: bool = True) -> None:
        if verbose:
            console.section(title="Loading tabular data")
            logger.info(f"Loading from: {self._filepath}")

        result = {"dataframe": pd.read_csv(self._filepath)}

        if verbose:
            logger.info(
                f"Finished loading tabular data. Dataframe shape: {result['dataframe'].shape}"
            )

        return result


class LazyAudioDataset(Dataset):
    '''
    LazyLoader dataset class, takes in a data directory and a target sampling rate.
    Used to lazyload audio files since the dataset is large
    '''
    def __init__(
        self,
        data_dir: str,
        target_sample_rate: int = 16000,
    ):
        self.file_paths = list(Path(data_dir).glob("*.wav"))
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filepath = self.file_paths[idx]

        waveform_np, sample_rate = sf.read(filepath, dtype="float32")

        waveform = torch.from_numpy(waveform_np)

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.transpose(0, 1)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)

        return waveform


class AudioDataLoaderModule(ModelPipelineStep):
    name = "AudioDataLoader"
    inputs = {}
    outputs = {"dataframe"}

    def __init__(
        self,
        filepath: Path = Path(__file__).parent.parent
        / "/sound_files/sound_files", 
    ) -> None:
        self._filepath = filepath

        super().__init__()

    def run(self, verbose: bool = True):
        if verbose:
            console.section(title="Loading audio data lazily")
            logger.info(f"Loading from: {self._filepath}")

        # Initialize the custom dataset
        dataset = LazyAudioDataset(data_dir=self._filepath, target_sample_rate=16000)

        if verbose:
            logger.info(f"Finished scanning folder. Found {len(dataset)} audio files.")

        # Return it directly for the pipeline to pass to the PyTorch DataLoader later
        return {"dataset": dataset}
