import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from logger import console, logger
from modules.lstm.lstm_model import FusionLSTMModel
from modules.model_bases import TrainerBase
from torch_datasets.lstm_dataset import LSTMBeeAudioDataset, pad_collate_fn


class FusionLSTMTrainModule(TrainerBase):
    """
    Pipeline step that trains a fused LSTM and MLP model on audio and tabular features.
    """

    name = "FusionLSTMTrainModule"
    inputs = {"x_train", "y_train", "x_val", "y_val"}
    outputs = {"model"}

    def __init__(
        self,
        drop_column: list[str],
        embeddings_model: tuple,
        lstm_layers: tuple,
        classification_hidden_size: int,
        audio_dir: Path,
        n_mfcc: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float = 1e-4,
        patience: int | None = 5,
        num_workers: int = 0,
        seed: int = 42,
        device: str | None = None,
        save_path: Path = Path(__file__).parents[3] / "trained_models" / "lstm",
    ) -> None:
        """
        Initialize the training module for the fused LSTM and MLP model.

        Args:
            drop_column (list[str]): Columns to drop from the dataset.
            layers (tuple): Number of hidden units in each MLP layer.
            lstm_layers (tuple): Number of hidden units in each LSTM layer.
            classification_hidden_size (int): Number of hidden units in the classification layer.
            audio_dir (Path): Path to the directory containing the audio files.
            n_mfcc (int): Number of MFCC features to extract from the audio files.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size to use during training.
            learning_rate (float): Learning rate to use during training.
            weight_decay (float): L2 regularization strength for Adam, to curb the
                overfitting seen on this small dataset. Defaults to 1e-4.
            patience (int | None): Early-stopping patience in epochs. Training stops when
                the validation loss has not improved for this many epochs (the best
                checkpoint is restored regardless). None disables early stopping.
                Defaults to 5.
            num_workers (int): Number of workers for data loading. Defaults to 0.
            seed (int): Random seed for reproducibility. Defaults to 42.
            device (str | None): Device to use for training. Defaults to None.
            save_path (Path): Path to save the trained model. Defaults to the default save path.

        Returns:
            None
        """
        super().__init__()
        self._drop_column = drop_column
        self._embeddings_model = embeddings_model
        self._lstm_layers = lstm_layers
        self._classification_hidden_size = classification_hidden_size
        self._audio_dir = audio_dir
        self._n_mfcc = n_mfcc

        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._patience = patience
        self._num_workers = num_workers
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
            DataLoader: Dataloader yielding (audio_features, tabular_features, labels) batches.
        """
        dataset = LSTMBeeAudioDataset(
            features=x,
            labels=y,
            audio_dir=self._audio_dir,
            n_mfcc=self._n_mfcc,
            columns_to_drop=self._drop_column,
        )
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=(self._device.type == "cuda"),
            collate_fn=pad_collate_fn,
        )

    def _save_model(self, model: FusionLSTMModel) -> None:
        """
        Save the fused LSTM and MLP model.

        Args:
            model (FusionLSTMModel): The fused LSTM and MLP model.

        Returns:
            None
        """
        if not self._save_path.exists():
            self._save_path.mkdir(parents=True)

        nr = (
            len(
                [
                    f
                    for f in self._save_path.iterdir()
                    if f.name.startswith("fusionlstm")
                ]
            )
            + 1
        )
        torch.save(model.state_dict(), self._save_path / f"fusionlstm_{nr}.pth")

    def _train_loop(
        self,
        model: FusionLSTMModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimiser: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        verbose: bool = True,
    ) -> None:
        """
        Run the full training loop over all epochs.

        Args:
            model (FusionLSTMModel): The model to train.
            train_loader (DataLoader): The data loader for training.
            val_loader (DataLoader): The data loader for validation.
            criterion (nn.Module): The loss function.
            optimiser (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler.LRScheduler): The learning rate scheduler.
            verbose (bool): Whether to log training progress. Defaults to True.

        Returns:
            None
        """
        best_val_loss = float("inf")
        best_state: dict | None = None
        best_epoch: int | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self._epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            if verbose:
                logger.info(f"Epoch {epoch}/{self._epochs} - Training...")

            for (
                batch_features,
                batch_tabular_features,
                batch_labels,
                _,
            ) in train_loader:
                batch_features = batch_features.to(self._device)
                batch_tabular_features = batch_tabular_features.to(self._device)
                batch_labels = batch_labels.to(self._device)

                optimiser.zero_grad()
                logits = model(batch_features, batch_tabular_features)
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
                val_loss, val_acc, pred_zeros, pred_ones, actual_zeros, actual_ones = (
                    self._evaluate(model, val_loader, criterion)
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if verbose:
                    logger.info(
                        f"Epoch completed [{epoch:>3}/{self._epochs}]  "
                        f"loss={epoch_loss:.4f}  acc={epoch_acc:.4f}    "
                        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  |  "
                        f"pred 0={pred_zeros} 1={pred_ones}  actual 0={actual_zeros} 1={actual_ones}"
                    )

                if (
                    self._patience is not None
                    and epochs_without_improvement >= self._patience
                ):
                    if verbose:
                        logger.info(
                            f"Early stopping at epoch {epoch}: no val_loss improvement "
                            f"for {self._patience} epoch(s) (best epoch {best_epoch})."
                        )
                    break
            elif verbose:
                logger.info(
                    f"Epoch completed [{epoch:>3}/{self._epochs}]  "
                    f"loss={epoch_loss:.4f}  acc={epoch_acc:.4f}"
                )

        # Restore the best-val-loss weights (if validation was performed).
        if best_state is not None:
            model.load_state_dict(best_state)
            if verbose:
                logger.info(
                    f"Restored best checkpoint from epoch {best_epoch} "
                    f"(val_loss={best_val_loss:.4f})."
                )

    @torch.no_grad()
    def _evaluate(
        self, model: FusionLSTMModel, loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        """
        Run a single evaluation pass over a loader without updating weights.

        Args:
            model (FusionLSTMModel): The model being evaluated.
            loader (DataLoader): Validation data loader.
            criterion (nn.Module): Loss function (same as training).

        Returns:
            tuple[float, float]: Average loss and accuracy over the validation set.
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        pred_zeros = 0
        pred_ones = 0
        actual_zeros = 0
        actual_ones = 0

        for (
            batch_features,
            batch_tabular_features,
            batch_labels,
            batch_lengths,
        ) in loader:
            batch_features = batch_features.to(self._device)
            batch_tabular_features = batch_tabular_features.to(self._device)
            batch_labels = batch_labels.to(self._device)

            logits = model(batch_features, batch_tabular_features, batch_lengths)
            loss = criterion(logits.squeeze(1), batch_labels.float())

            running_loss += loss.item() * len(batch_labels)
            preds = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

            pred_zeros += (preds == 0).sum().item()
            pred_ones += (preds == 1).sum().item()
            actual_zeros += (batch_labels == 0).sum().item()
            actual_ones += (batch_labels == 1).sum().item()

        return (
            running_loss / total,
            correct / total,
            pred_zeros,
            pred_ones,
            actual_zeros,
            actual_ones,
        )

    def run(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        verbose: bool = True,
        save_model: bool = True,
    ) -> dict[str, FusionLSTMModel]:
        """
        Train the fused LSTM and MLP model.

        Args:
            x_train (pd.DataFrame): Training data.
            y_train (pd.Series): Training labels.
            x_val (pd.DataFrame): Validation data.
            y_val (pd.Series): Validation labels.
            verbose (bool): Whether to log training progress. Defaults to True.
            save_model (bool): Whether to persist the trained model to disk. Set to False
                during cross-validation so only the final refit produces a checkpoint.
                Defaults to True.

        Returns:
            dict[str, FusionLSTMModel]: A dictionary containing the trained model.
        """
        self._set_seed()

        if verbose:
            console.section(title="Training Fusion LSTM model")
            logger.info(
                f"Processing on: {x_train.shape[-1] - len(self._drop_column)} columns"
            )
            logger.info(f"Total training samples: {len(x_train):,}")
            logger.info(f"Total validation samples: {len(x_val):,}")
            logger.info("Building dataloaders...")

        train_loader = self._build_dataloader(x_train, y_train)
        val_loader = self._build_dataloader(x_val, y_val)

        model = FusionLSTMModel(
            lstm_layers=self._lstm_layers,
            features_input_size=self._embeddings_model[-1].out_features,
            ff_hidden_size=self._classification_hidden_size,
            embeddings_model=self._embeddings_model,
        ).to(self._device)

        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(
            model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self._epochs
        )

        if verbose:
            logger.info(f"Starting training for {self._epochs} epochs...")

        self._train_loop(
            model, train_loader, val_loader, criterion, optimiser, scheduler, verbose
        )

        if verbose:
            logger.info("Finished creating composite model.")

        if save_model:
            if verbose:
                logger.info("Saving model...")
            self._save_model(model)
            if verbose:
                logger.info("Model saved.")
        elif verbose:
            logger.info("Skipping checkpoint persistence (save_model=False).")

        return {"model": model}

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        save_model: bool = True,
    ) -> FusionLSTMModel:
        """
        Train and return the model without pipeline scaffolding.

        Args:
            x_train (pd.DataFrame): Training data.
            y_train (pd.Series): Training labels.
            x_val (pd.DataFrame): Validation data.
            y_val (pd.Series): Validation labels.
            save_model (bool): Whether to persist the trained model to disk. Defaults to True.

        Returns:
            FusionLSTMModel: The trained model.
        """
        return self.run(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            verbose=True,
            save_model=save_model,
        )["model"]
