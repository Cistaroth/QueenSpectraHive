import torch
import torchaudio.transforms as T
from typing import Any

from pipeline import ModelPipelineStep
from logger import console, logger


class MFCCExtractorModule(ModelPipelineStep):
    name = "MFCC Extractor"
    
    inputs = {"waveform", "sample_rate"}
    outputs = {"mfcc"}

    def __init__(
        self,
        n_mfcc: int = 40,
        sample_rate: int = 16000
    ) -> None:
        """
        Initializes the MFCCExtractor class.

        Args:
            n_mfcc (int, optional): Number of MFCCs to retain. Defaults to 40.
            sample_rate (int, optional): Sample rate of the audio. Defaults to 16000.
        
        Returns:
            None
        """
        super().__init__()

        self._n_mfcc = n_mfcc

        self._mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
        )

    def run(self, waveform: torch.Tensor, verbose: bool = True) -> dict[str, Any]:
        """
        Extracts MFCC vectors from a raw waveform.

        Args:
            waveform (torch.Tensor): Raw audio tensor of shape [Channels, Time_Samples].
            verbose (bool, optional): Verbose logging. Defaults to True.
        Returns:
            dict[str, Any]: Dictionary containing the output 'mfcc' tensor.
        """
        if verbose:
            console.section("Extracting MFCC Audio Vectors")

        try:
            mfcc_data = self._mfcc_transform(waveform)

            if verbose:
                logger.info("Successfully processed segment.")
                logger.info(f"Generated MFCC array shape: {mfcc_data.shape}")

            return {"mfcc": mfcc_data}

        except Exception as e:
            logger.error(f"Failed to calculate MFCC coefficients: {e}")
            return {"mfcc": torch.zeros((waveform.shape[0], self._n_mfcc, 1))}
