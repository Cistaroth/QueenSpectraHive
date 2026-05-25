from pathlib import Path

import pandas as pd

from pipeline import ModelPipelineStep
from logger import console, logger

class TabularDataLoaderModule(ModelPipelineStep):
    name = "TabularDataLoader"
    inputs = set()
    outputs = {"dataframe"}

    def __init__(
        self,
        filepath: Path = (
            Path(__file__).parent.parent / "data" / "all_data_updated.csv"
        ),
        max_samples: int | None = None,
    ) -> None:
        """
        Initializes the TabularDataLoader class

        Args:
            filepath (Path, optional): Filepath to the tabular data.
                Defaults to Path(__file__).parent / "data" /
                "all_data_updated.csv".
            max_samples (int | None, optional): Maximum number of samples to load.
                If None, loads all data. Defaults to None.
        Returns:
            None
        """
        self._filepath = filepath
        self._max_samples = max_samples

        super().__init__()

    def run(
        self,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Loads the tabular data

        Args:
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: Dictionary containing the loaded dataframe.
        """
        if verbose:
            console.section(title="Loading tabular data")
            logger.info(f"Loading from: {self._filepath}")

        df = pd.read_csv(self._filepath)
        
        # Apply max_samples limit if specified
        if self._max_samples is not None:
            df = df.head(self._max_samples)
            if verbose:
                logger.info(f"Limited to {self._max_samples} samples")

        result = {"dataframe": df}

        if verbose:
            logger.info(
                "Finished loading tabular data. "
                f"Dataframe shape: {result['dataframe'].shape}"
            )

        return result