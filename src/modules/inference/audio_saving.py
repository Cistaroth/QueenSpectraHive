import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
from fastapi import UploadFile

from pipeline import ModelPipelineStep
from logger import console, logger

class AudioSavingModule(ModelPipelineStep):
    name = "AudioSavingModule"
    inputs = {"file"}
    outputs = {"result", "x_test"}

    def __init__(
        self,
        save_path: Path = Path(__file__).parents[2] / "data" / "sound_files" / "sound_files",
        chunk_duration: float = 15.0,
    ) -> None:
        """
        Initialize the AudioSavingModule.

        Args:
            save_path (Path): The path to save the audio file.
            chunk_duration (float): Length in seconds of the window to feed the model.
                Must match the chunk_duration used at training time (see
                AudioSplicerModule). The dataframe returned to the inference pipeline
                will carry a centered (start_sec, end_sec) window of this length so
                LazyAudioDataset crops the same way training did, instead of letting
                ASTFeatureExtractor silently truncate to its default ~10.24 s prefix.

        Returns:
            None
        """
        super().__init__()

        self._save_path = save_path
        self._chunk_duration = float(chunk_duration)

    def run(
        self,
        file: UploadFile,
        verbose: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Save the audio file to a specified path.

        Args:
            file (UploadFile): The audio file to be saved.

        Returns:
            dict[str, pd.DataFrame]: A dictionary containing the saved file.
        """
        if verbose:
            console.section(title="Saving audio file")
            console.print(f"Saving audio file to {self._save_path}")

        # Save the audio file to a specified path
        save_path = self._save_path / "inference_input__segment.wav"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        if verbose:
            console.print(f"Audio file saved to {save_path}")

        # Inspect the saved file to compute a centered chunk_duration-second
        # crop window. This matches how training data is cropped by
        # AudioSplicerModule (random chunk_duration window), so the model sees
        # the same kind of window at inference time instead of just the first
        # ~10 s left over after ASTFeatureExtractor truncation.
        try:
            sound_info = sf.info(save_path)
            duration_sec = sound_info.frames / sound_info.samplerate
        except Exception as e:
            logger.warning(
                f"Could not read duration of saved audio at {save_path}: {e}. "
                "Falling back to a full-file crop."
            )
            duration_sec = self._chunk_duration

        if duration_sec <= self._chunk_duration:
            start_sec = 0.0
            end_sec = float(duration_sec)
        else:
            start_sec = (duration_sec - self._chunk_duration) / 2.0
            end_sec = start_sec + self._chunk_duration

        if verbose:
            logger.info(
                f"Audio duration: {duration_sec:.2f}s; "
                f"centered crop -> start_sec={start_sec:.2f}s, end_sec={end_sec:.2f}s "
                f"(chunk_duration={self._chunk_duration:.2f}s)"
            )

        # Return the saved file together with the centered crop window so the
        # downstream LazyAudioDataset can slice exactly that range.
        dataframe = pd.DataFrame(
            {
                "file name": ["inference_input.raw"],
                "start_sec": [start_sec],
                "end_sec": [end_sec],
            }
        )

        return {
            "result": True,
            "x_test": dataframe
        }

