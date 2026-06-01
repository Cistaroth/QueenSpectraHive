from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from pipeline import ModelPipelineStep
from modules.data_augmentation.audio_splicing import random_audio_slice
from logger import console, logger

AUDIO_PATH_COLUMN = "file name"
SLICE_COLUMNS = ["start_sec", "end_sec"]
AUDIO_METADATA_COLUMNS = ["file name", "start_sec", "end_sec"]


class TabularSMOTE(ModelPipelineStep):
    name = "TabularSMOTE"

    inputs = {"x_train", "y_train"}
    outputs = {"x_train", "y_train"}

    def __init__(
        self,
        seed: int = 69,
        audio_metadata_columns: list[str] | None = None,
        audio_dir: Path | str | None = None,
        chunk_duration: int = 15,
    ) -> None:
        """
        Initializes the TabularSMOTE class.

        Args:
            seed (int, optional): Random state seed for SMOTE replication. Defaults to 69.
            audio_metadata_columns (list[str] | None, optional): Columns that carry audio
                metadata (path + slice bounds) rather than model features. These are held
                out of the SMOTE interpolation and, for synthetic rows, borrowed from a
                real same-class row so the audio modality stays valid. Defaults to
                ["file name", "start_sec", "end_sec"].
            audio_dir (Path | str | None, optional): Directory holding the audio segment
                files. When provided, each synthetic row is given a freshly sampled random
                slice window of its borrowed clip (via the audio splicer's frame sampler)
                instead of copying the donor row's window. Defaults to None.
            chunk_duration (int, optional): Length in seconds of the random slice window
                drawn for synthetic rows. Defaults to 15.
        Returns:
            None
        """
        super().__init__()

        self._random_state = seed
        self._sampler = SMOTE(random_state=self._random_state)
        self._audio_metadata_columns = (
            list(AUDIO_METADATA_COLUMNS)
            if audio_metadata_columns is None
            else list(audio_metadata_columns)
        )
        self._audio_dir = Path(audio_dir) if audio_dir is not None else None
        self._chunk_duration = chunk_duration

    def _reattach_audio_metadata(
        self,
        x_original: pd.DataFrame,
        y_original: pd.Series,
        x_resampled: pd.DataFrame,
        y_resampled: pd.Series,
        metadata_cols: list[str],
    ) -> pd.DataFrame:
        """
        Restore the audio metadata columns onto the resampled feature frame.

        Args:
            x_original (pd.DataFrame): Features before resampling (with metadata cols).
            y_original (pd.Series): Labels before resampling.
            x_resampled (pd.DataFrame): Resampled numeric features (no metadata cols yet).
            y_resampled (pd.Series): Labels after resampling.
            metadata_cols (list[str]): Metadata columns present in x_original.
        Returns:
            pd.DataFrame: ``x_resampled`` with the metadata columns reattached, in the
                original column order.
        """
        n_total = len(x_resampled)
        n_original = len(x_original)

        source_idx = np.empty(n_total, dtype=int)
        source_idx[:n_original] = np.arange(n_original)

        if n_total > n_original:
            rng = np.random.default_rng(self._random_state)
            y_orig = y_original.to_numpy()
            class_to_rows = {
                label: np.flatnonzero(y_orig == label) for label in np.unique(y_orig)
            }
            synthetic_labels = y_resampled.to_numpy()[n_original:]
            for offset, label in enumerate(synthetic_labels):
                source_idx[n_original + offset] = rng.choice(class_to_rows[label])

        metadata = x_original[metadata_cols].reset_index(drop=True)
        borrowed = metadata.iloc[source_idx].reset_index(drop=True)
        for col in metadata_cols:
            x_resampled[col] = borrowed[col].values

        can_slice = (
            self._audio_dir is not None
            and n_total > n_original
            and AUDIO_PATH_COLUMN in metadata_cols
            and set(SLICE_COLUMNS).issubset(metadata_cols)
        )
        if can_slice:
            rng = np.random.default_rng(self._random_state + 1)
            for i in range(n_original, n_total):
                start, end = random_audio_slice(
                    x_resampled.at[i, AUDIO_PATH_COLUMN],
                    self._audio_dir,
                    self._chunk_duration,
                    rng=rng,
                )
                x_resampled.at[i, "start_sec"] = start
                x_resampled.at[i, "end_sec"] = end

        return x_resampled[x_original.columns]

    def run(
        self, x_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True
    ) -> dict[str, Any]:
        """
        Runs SMOTE resampling over training features.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target labels.
            verbose (bool, optional): Verbose logging mode. Defaults to True.
        Returns:
            dict[str, Any]: Dictionary containing balanced train sets.
        """

        if verbose:
            console.section("Handling Class Imbalance on Tabular Data")
            logger.info(f"Original class distribution: \n{y_train.value_counts()}")

        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        metadata_cols = [
            c for c in self._audio_metadata_columns if c in x_train.columns
        ]
        x_numeric = x_train.drop(columns=metadata_cols)

        x_resampled, y_resampled = cast(
            "tuple[pd.DataFrame, pd.Series]",
            self._sampler.fit_resample(x_numeric, y_train),
        )
        x_resampled = pd.DataFrame(x_resampled, columns=x_numeric.columns).reset_index(
            drop=True
        )
        y_resampled = pd.Series(
            cast("np.ndarray | pd.Series", y_resampled), name=y_train.name
        ).reset_index(drop=True)

        if metadata_cols:
            x_resampled = self._reattach_audio_metadata(
                x_train, y_train, x_resampled, y_resampled, metadata_cols
            )

        if verbose:
            logger.info(
                f"Finished resampling of tabular data. \n"
                f"New class distribution: \n: {y_resampled.value_counts()}"
            )
        return {
            "x_train": x_resampled,
            "y_train": y_resampled,
        }
