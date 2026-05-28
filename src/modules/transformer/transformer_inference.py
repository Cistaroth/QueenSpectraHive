from pathlib import Path

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from modules.model_bases import InferencerBase
from logger import console, logger

from modules.transformer.transformer_train import AudioTransformer
from torch_datasets.transformer_dataset import TransformerBeeAudioDataset


class TransformerInferenceModule(InferencerBase):
    """
    Pipeline step that runs inference with a fine-tuned Audio Spectrogram Transformer.
  
    """

    name = "TransformerInference"
    inputs = {"model", "x_test"}
    outputs = {"y_pred", "y_pred_proba"}

    def __init__(
        self,
        audio_dir: str = "data/sound_files/sound_files",
        audio_path_col: str = "file name",
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        batch_size: int = 32,
        num_workers: int = 0,
        device: str | None = None,
    ) -> None:
        """
        Initialise the transformer inference step.

        Args:
            audio_dir (str)         : Directory containing audio files.
            audio_path_col (str)    : Column in x_test holding the audio file path.
            pretrained_model (str)  : Hugging Face model ID for AST feature extractor.
            batch_size (int)        : DataLoader batch size.
            num_workers (int)       : DataLoader worker processes.
            device (str | None)     : 'cpu', 'cuda', 'mps', or None (auto-detect).
        Returns:
            None
        """
        super().__init__()

        self._audio_dir = Path(audio_dir)
        self._audio_col = audio_path_col
        self._pretrained_model = pretrained_model
        self._batch_size = batch_size
        self._num_workers = num_workers

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

    
    def _build_loader(self, x_test: pd.DataFrame) -> DataLoader:
        """
        Build the dataloader for inference.

        Args:
            x_test (pd.DataFrame) : Test features with audio metadata.
        Returns:
            DataLoader
        """
        dataset = TransformerBeeAudioDataset(
            features=x_test,
            audio_dir=self._audio_dir,
            pretrained_model=self._pretrained_model,
        )
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=(self._device.type == "cuda"),
        )

    def run(
        self,
        model: AudioTransformer,
        x_test: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Run batch inference and return predictions + probabilities.

        Args:
            model   (AudioTransformer): The fine-tuned transformer model.
            x_test  (pd.DataFrame)    : Test features with audio metadata.
            verbose (bool)            : Verbose logging. Defaults to True.
        Returns:
            dict containing:
                y_pred       - predicted class indices  
                y_pred_proba - probability of class 1   
        """
        if verbose:
            console.section("Evaluating Audio Spectrogram Transformer")
            logger.info(f"Device     : {self._device}")
            logger.info(f"Test shape : {x_test.shape}")

        model = model.to(self._device)
        model.eval()

        loader = self._build_loader(x_test)

        all_preds: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []

        with torch.no_grad():
            for batch_mel, _ in loader:
                batch_mel = batch_mel.to(self._device)
                logits = model(batch_mel)     
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)  

                all_preds.append(preds.cpu().numpy())
                all_proba.append(probs[:, 1].cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_pred_proba = np.concatenate(all_proba)

        if verbose:
            logger.info(
                f"Finished inference. Prediction shape: {y_pred.shape}"
            )

        return {
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

    def inference(
        self,
        model: AudioTransformer,
        x_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return predicted class labels for x_test.

        Args:
            model  (AudioTransformer)
            x_test (pd.DataFrame)   
        Returns:
            np.ndarray: Predicted class indices
        """
        return self.run(model=model, x_test=x_test, verbose=False)[
            "y_pred"
        ]

    def inference_proba(
        self,
        model: AudioTransformer,
        x_test: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return predicted probabilities (positive class) for x_test.

        Args:
            model  (AudioTransformer)
            x_test (pd.DataFrame)    
        Returns:
            np.ndarray: Probability of class 1, shape (N,).
        """
        return self.run(model=model, x_test=x_test, verbose=False)[
            "y_pred_proba"
        ]