from pathlib import Path

import pandas as pd

from pipeline import ModelPipelineStep
from logger import console, logger

class TabularDataLoaderModule(ModelPipelineStep):
    name = "TabularDataLoader"
    inputs = {}
    outputs = {"dataframe"}

    def __init__(
        self,
        filepath: Path = Path(__file__).parent.parent / "data" / "all_data_updated.csv",
    ) -> None:
        self._filepath = filepath

        super().__init__()
    
    def run(
        self,
        verbose: bool = True
    )-> None:
        if verbose:
            console.section(title="Loading tabular data")
            logger.info(f"Loading from: {self._filepath}")

        result = {
            "dataframe": pd.read_csv(self._filepath)
        }

        if verbose:
            logger.info(f"Finished loading tabular data. Dataframe shape: {result['dataframe'].shape}")

        return result
