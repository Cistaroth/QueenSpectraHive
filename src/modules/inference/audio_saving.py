import shutil
from pathlib import Path

import pandas as pd
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
    ) -> None:
        """
        Initialize the AudioSavingModule.

        Args:
            save_path (Path): The path to save the audio file.
        
        Returns:
            None
        """
        super().__init__()

        self._save_path = save_path

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

        # Return the saved file
        dataframe = pd.DataFrame(
            {
                "file name": ["inference_input.raw"],
            }     
        )

        return {
            "result": True,
            "x_test": dataframe
        }

