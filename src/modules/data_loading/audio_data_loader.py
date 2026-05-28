from pathlib import Path
import pandas as pd

from pipeline import ModelPipelineStep
from torch_datasets.lazy_audio_dataset import LazyAudioDataset
from logger import console, logger

class AudioDataLoaderModule(ModelPipelineStep):
    name = "AudioDataLoader"
    inputs = {"dataframe"}
    outputs = {"dataset"}

    def __init__(
        self,
        filepath: Path = Path(__file__).parent.parent,
        sample_rate: int = 16000
    ) -> None:
        self._filepath = filepath
        self.sample_rate = sample_rate

        super().__init__()

    def run(self,  dataframe: pd.DataFrame, verbose: bool = True):
        if verbose:
            console.section(title="Loading audio data lazily")
            logger.info(f"Loading from: {self._filepath}")

        # Initialize the custom dataset
        dataset = LazyAudioDataset(df=dataframe,audio_dir=self._filepath, target_sample_rate=self.sample_rate)

        if verbose:
            logger.info(f"Finished scanning folder. Found {len(dataset)} audio files.")

        # Return it directly for the pipeline to pass to the PyTorch DataLoader later
        return {"dataset": dataset}
