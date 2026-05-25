from typing import Any, cast

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from pipeline import ModelPipelineStep
from logger import console, logger


class TabularSMOTE(ModelPipelineStep):
    name = "TabularSMOTE"

    inputs = {"x_train", "y_train"}
    outputs = {"x_train", "y_train"}

    def __init__(self, seed: int = 69) -> None:
        """
        Initializes the TabularInterpolation class.

        Args:
            seed (int, optional): Random state seed for SMOTE replication. Defaults to 69.
        Returns:
            None
        """
        super().__init__()

        self._random_state = seed
        self._sampler = SMOTE(random_state=self._random_state)

    def run(
        self, x_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True
    ) -> dict[str, Any]:
        """
        Runs SMOTE resampling over numeric training features.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target labels.
            verbose (bool, optional): Verbose logging mode. Defaults to True.
        Returns:
            dict[str, Any]: Dictionary containing balanced train sets and untouched test sets.
        """

        if verbose:
            console.section("Handling Class Imbalance on Tabular Data")
            logger.info(f"Original class distribution: \n{y_train.value_counts()}")

        x_numeric = x_train.copy()
        audio_paths: pd.Series | None = None
        if "file name" in x_train.columns:
            audio_paths = x_train["file name"].copy()
            x_numeric = x_train.drop(columns=["file name"])

        x_resampled, y_resampled = cast(
            "tuple[pd.DataFrame, pd.Series]",
            self._sampler.fit_resample(x_numeric, y_train)
        )
        x_resampled = pd.DataFrame(x_resampled, columns=x_numeric.columns)

        if "file name" in x_train.columns and audio_paths is not None:
            x_resampled["file name"] = audio_paths.reset_index(drop=True)

        y_resampled = pd.Series(
            cast("np.ndarray | pd.Series", y_resampled), name=y_train.name
        )

        if verbose:
            logger.info(
                f"Finised resampling of tabular data. \n"
                f"New class distribution: \n: {y_resampled.value_counts()}"
            )
        return {
            "x_train": x_resampled,
            "y_train": y_resampled,
        }
