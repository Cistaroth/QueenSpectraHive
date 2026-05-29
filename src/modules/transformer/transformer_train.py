import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.model_bases import TrainerBase
from modules.transformer.transformer_model import AudioTransformer
from torch_datasets.transformer_dataset import TransformerBeeAudioDataset

from logger import console, logger


class TransformerTrainModule(TrainerBase):
    """
    Pipeline step that fine-tunes a pretrained Audio Spectrogram Transformer.
    """

    name = "TransformerTraining"
    inputs = {"x_train", "y_train", "x_val", "y_val"}
    outputs = {"model"}

    def __init__(
        self,
        audio_dir: Path | str = "data/sound_files/sound_files",
        num_classes: int = 1,
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_workers: int = 0,
        audio_path_col: str = "path",
        device: str | None = None,
        seed: int = 42,
        save_path: Path | None = Path(__file__).parents[3] / "trained_models" / "transformer",
    ) -> None:
        """
        Initialize the step.

        Args:
            audio_dir (Path | str): Path to the directory containing the audio files.
            num_classes (int): Number of output units of the classification head.
            pretrained_model (str): Hugging Face model ID for the AST feature extractor.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            num_workers (int): Number of workers for data loading.
            audio_path_col (str): Name of the column containing the audio file paths.
            device (str | None): Device to train on. Defaults to None.
            seed (int): Random seed for reproducibility. Defaults to 42.
            save_path (Path | None): Path to save the fine-tuned model. Defaults to the default save path.

        Returns:
            None
        """
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
        self._save_path = save_path

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
        """
        Set the random seed for reproducibility.

        Returns:
            None
        """
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(self._seed)

    def _build_dataloader(self, x: pd.DataFrame, y: pd.Series) -> DataLoader:
        """
        Build a dataloader from a feature frame and its labels.

        Args:
            x (pd.DataFrame): Feature frame.
            y (pd.Series): Labels.

        Returns:
            DataLoader: Dataloader yielding (features, labels) batches.
        """
        dataset = TransformerBeeAudioDataset(
            features=x,
            labels=y,
            audio_dir=self._audio_dir,
            pretrained_model=self._pretrained_model,
        )
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=(self._device.type == "cuda"),
        )

    def _save_model(self, model: AudioTransformer) -> None:
        """
        Save the fine-tuned model.

        Args:
            model (AudioTransformer): The fine-tuned model.

        Returns:
            None
        """
        if not self._save_path.exists():
            self._save_path.mkdir(parents=True)

        nr = len([f for f in self._save_path.iterdir() if f.name.startswith("transformer")]) + 1
        torch.save(model.state_dict(), self._save_path / f"transformer_{nr}.pth")

    def _train_loop(
        self,
        model: AudioTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        verbose: bool = True,
    ) -> None:
        """
        Run the full training loop over all epochs.

        Args:
            model (AudioTransformer): The model to train.
            train_loader (DataLoader): The data loader for training.
            val_loader (DataLoader | None): The data loader for validation. If None,
                validation is skipped.
            criterion (nn.Module): The loss function.
            optimiser (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
            verbose (bool): Whether to log training progress. Defaults to True.

        Returns:
            None
        """
        for epoch in range(1, self._epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            if verbose:
                logger.info(f"Epoch {epoch}/{self._epochs} - Training...")

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self._device)
                batch_labels = batch_labels.to(self._device)

                optimiser.zero_grad()
                logits = model(batch_features)
                loss = criterion(logits.squeeze(1), batch_labels.float())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

                running_loss += loss.item() * len(batch_labels)
                preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

            scheduler.step()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            if val_loader is not None:
                val_loss, val_acc = self._evaluate(model, val_loader, criterion)
                if verbose:
                    logger.info(
                        f"Epoch completed [{epoch:>3}/{self._epochs}]  "
                        f"loss={epoch_loss:.4f}  acc={epoch_acc:.4f}    "
                        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                    )
            elif verbose:
                logger.info(
                    f"Epoch completed [{epoch:>3}/{self._epochs}]  "
                    f"loss={epoch_loss:.4f}  acc={epoch_acc:.4f}"
                )

    @torch.no_grad()
    def _evaluate(self, model: AudioTransformer, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        """
        Run a single evaluation pass over a loader without updating weights.

        Args:
            model (AudioTransformer): The model being evaluated.
            loader (DataLoader): Validation data loader.
            criterion (nn.Module): Loss function (same as training).

        Returns:
            tuple[float, float]: Average loss and accuracy over the validation set.
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(self._device)
            batch_labels = batch_labels.to(self._device)

            logits = model(batch_features)
            loss = criterion(logits.squeeze(1), batch_labels.float())

            running_loss += loss.item() * len(batch_labels)
            preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

        return running_loss / total, correct / total

    def run(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        verbose: bool = True,
    ) -> dict[str, AudioTransformer]:
        """
        Fine-tune the pretrained AudioTransformer.

        Args:
            x_train (pd.DataFrame): Training data.
            y_train (pd.Series): Training labels.
            x_val (pd.DataFrame | None): Validation data. If None, validation is skipped.
            y_val (pd.Series | None): Validation labels. If None, validation is skipped.
            verbose (bool): Whether to log training progress. Defaults to True.

        Returns:
            dict[str, AudioTransformer]: A dictionary containing the fine-tuned model.
        """
        self._set_seed()

        has_val = x_val is not None and y_val is not None

        if verbose:
            console.section("Pretrained Audio Spectrogram Transformer")
            logger.info(f"Pretrained Model: {self._pretrained_model}")
            logger.info(f"Total training samples: {len(x_train):,}")
            if has_val:
                logger.info(f"Total validation samples: {len(x_val):,}")
            else:
                logger.info("No validation data provided - validation will be skipped.")
            logger.info("Building dataloaders...")

        train_loader = self._build_dataloader(x_train, y_train)
        val_loader = self._build_dataloader(x_val, y_val) if has_val else None

        model = AudioTransformer(
            num_classes=self._num_classes,
            pretrained_model=self._pretrained_model,
        ).to(self._device)

        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self._epochs)

        if verbose:
            logger.info(f"Starting fine-tuning for {self._epochs} epochs...")

        self._train_loop(model, train_loader, val_loader, criterion, optimiser, scheduler, verbose)

        if verbose:
            logger.info("Finished fine-tuning Audio Spectrogram Transformer.")
            logger.info("Saving model...")

        self._save_model(model)

        if verbose:
            logger.info("Model saved.")

        return {"model": model}

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> AudioTransformer:
        """
        Fine-tune and return the model without pipeline scaffolding.

        Args:
            x_train (pd.DataFrame): Training data.
            y_train (pd.Series): Training labels.
            x_val (pd.DataFrame | None): Validation data. If None, validation is skipped.
            y_val (pd.Series | None): Validation labels. If None, validation is skipped.

        Returns:
            AudioTransformer: The fine-tuned model.
        """
        return self.run(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, verbose=True)["model"]