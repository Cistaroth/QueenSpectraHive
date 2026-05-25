import torch
import torchaudio.transforms as T
from typing import Any

from pipeline import ModelPipelineStep
from logger import console, logger


class MFCCExtractorModule(ModelPipelineStep):
    """
    Slices raw PyTorch waveforms by physical time boundaries 
    and extracts Mel-Frequency Cepstral Coefficients (MFCCs).
    """

    name = "MFCC Extractor"

    # Accepting the raw waveform along with your tracking metadata markers
    inputs = {"waveform", "sample_rate"}
    outputs = {"mfcc"}

    def __init__(self, n_mfcc: int = 40) -> None:
        """
        Initializes the MFCCExtractor class.

        Args:
            n_mfcc (int, optional): Number of Mel-Frequency Cepstral Coefficients to retain. Defaults to 40.
        """
        super().__init__()
        self._n_mfcc = n_mfcc

    def run(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        verbose: bool = True
    ) -> dict[str, Any]:
        """
        Crops an input waveform using temporal second markers and extracts MFCC vectors.

        Args:
            waveform (torch.Tensor): Raw audio input stream tensor of shape [Channels, Time_Samples].
            sample_rate (int): The current sampling frequency of the provided waveform.
            verbose (bool, optional): Verbose logging execution mode. Defaults to True.
        Returns:
            dict[str, Any]: Dictionary containing the output 'mfcc' tensor.
        """
        if verbose:
            console.section("Extracting MFCC Audio Vectors")

        try:
            mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=self._n_mfcc,
                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64}
            )

            mfcc_data = mfcc_transform(waveform)

            if verbose:
                logger.info("Successfully processed segment.")
                logger.info(f"Generated MFCC array shape: {mfcc_data.shape}")

            return {"mfcc": mfcc_data}

        except Exception as e:
            logger.error(f"Failed to calculate MFCC coefficients: {e}")
            return {"mfcc": torch.zeros((waveform.shape[0], self._n_mfcc, 1))}