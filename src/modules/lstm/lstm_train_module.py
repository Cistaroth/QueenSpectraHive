import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification, ASTFeatureExtractor

from modules.model_bases import TrainerBase, TransformBase
from modules.data_loading.audio_data_loader import LazyAudioDataset, AudioDataLoaderModule

# Removed MFCCExtractorModule import
from logger import console, logger

from modules.data_loading.lazy_bee_audio_dataset import BeeAudioDataset


class AudioPassthroughLstm(TransformBase):
    """
    Identity transformer required by FineTuningConfiguration.

    The audio pipeline needs no tabular scaling; this no-op satisfies the
    interface so HyperparameterTuningStratifiedKFoldModule can call
    cfg.transformer.__name__ and transformer.fit_transform() without error.
    """

    name = "AudioPassthroughLstm"
    inputs: set[str] = set()
    outputs: set[str] = set()

    def run(self, *args: Any, **kwargs: Any) -> None:  # noqa: D102
        return None

    def fit_transform(self, x: Any) -> Any:  # noqa: D102
        return x

    def transform(self, x: Any) -> Any:  # noqa: D102
        return x



class AudioTransformer(nn.Module):
    """

    This model loads a pretrained AST checkpoint from Hugging Face and adapts it
    for two classes for on bee audio data.
    """

    def __init__(
        self,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.ast = ASTForAudioClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pretrained AST model.

        Args:
            x (torch.Tensor): Log-mel features of shape (B, time_frames, num_mel_bins).
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        outputs = self.ast(x)
        return outputs.logits


class FusionModelTrainModule(TrainerBase):
    """
    Pipeline step that fine-tunes a pretrained Audio Spectrogram Transformer.
    """

    name = "FusionModelTrain"
    inputs = {"x_train", "y_train", "fusion_model"}
    outputs = {"model"}

    def __init__(
        self,
        audio_dir: Path | str = "data/sound_files/sound_files",
        num_classes: int = 2,
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_workers: int = 0,
        audio_path_col: str = "path",
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self._audio_dir = Path(audio_dir)
        self._num_classes = num_classes
        self._pretrained_model = pretrained_model
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._num_workers = num_workers
        self._audio_path_col = audio_path_col
        self._seed = seed

        if device is None:
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

    def _set_seed(self) -> None:
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(self._seed)

    def run(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        verbose: bool = True,
    ) -> dict[str, AudioTransformer]:
        """
        Fine-tune the pretrained AudioTransformer.
        """
        self._set_seed()

        if verbose:
            console.section("Fine-tuning Pretrained Audio Spectrogram Transformer")
            logger.info(f"Samples: {len(x_train):,}  |  Classes: {self._num_classes}")
            logger.info(f"Pretrained Model: {self._pretrained_model}")
            print(f"\n[TransformerTraining]  Total training samples after splicing: {len(x_train):,}\n")

        dataset = BeeAudioDataset(
            df=x_train,
            labels=y_train,
            audio_dir=self._audio_dir,
            pretrained_model=self._pretrained_model,
        )
        loader = DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=(self._device.type == "cuda"),
        )

        # Imported model, optimiser, loss, and scheduler
        model = AudioTransformer(
            num_classes=self._num_classes,
            pretrained_model=self._pretrained_model,
        ).to(self._device)

        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self._epochs)
        if verbose:
            logger.info(f"Starting fine-tuning for {self._epochs} epochs...\n")

        for epoch in range(1, self._epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            if verbose:
                logger.info(f"Epoch {epoch}/{self._epochs} - Training...")

            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self._device)
                batch_labels = batch_labels.to(self._device)

                optimiser.zero_grad()
                logits = model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

                running_loss += loss.item() * len(batch_labels)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

            scheduler.step()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            if verbose:
                logger.info(f"Epoch completed [{epoch:>3}/{self._epochs}]  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

        if verbose:
            logger.info("Finished fine-tuning Audio Spectrogram Transformer.")

        return {"model": model}

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> AudioTransformer:
        """
        Fine-tune and return the model without pipeline scaffolding.
        """
        return self.run(x_train=x_train, y_train=y_train, verbose=True)["model"]
