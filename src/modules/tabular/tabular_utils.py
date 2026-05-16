import pandas as pd

from pipeline import ModelPipelineStep
from logger import console, logger


class TabularColumnDropperModule(ModelPipelineStep):
    name = "TabularColumnDropper"
    inputs = {"dataframe"}
    outputs = {"dataframe"}

    def __init__(
        self,
        drop_columns: list[str],
    ) -> None:
        """
        Initialize the tabular column dropper class

        Args:
            drop_columns (list[str]): The columns to drop
        Returns:
            None
        """
        self._drop_columns = drop_columns

        super().__init__()

    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Drop columns from the dataframe

        Args:
            dataframe (pd.DataFrame): The dataframe to drop columns from
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: The dataframe without the dropped columns
        """
        if verbose:
            console.section(title="Dropping columns")
            logger.info(f"Columns to drop: {self._drop_columns}")

        result = {
            "dataframe": dataframe.drop(columns=self._drop_columns),
        }

        if verbose:
            logger.info(
                "Finished dropping columns. "
                f"Dataframe shape: {result['dataframe'].shape}"
            )

        return result
    
class TabularOneHotEncoderModule(ModelPipelineStep):
    name = "OneHotEncoder"
    inputs = {"dataframe"}
    outputs = {"dataframe"}

    def __init__(
        self,
        columns_to_encode: list[str],
        drop_first: bool = True,
    ) -> None:
        """
        Initialize the one-hot encoder class

        Args:
            columns_to_encode (list[str]): The columns to encode
            drop_first (bool, optional): Whether to drop the first column.
                Defaults to True.
        Returns:
            None
        """
        self._columns_to_encode = columns_to_encode
        self._drop_first = drop_first

        super().__init__()

    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        One-hot encode columns in the dataframe
        
        Args:
            dataframe (pd.DataFrame): The dataframe to one-hot encode
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: The one-hot encoded dataframe
        """
        if verbose:
            console.section(title="One-hot encoding")
            logger.info(f"Columns to encode: {self._columns_to_encode}")

        result = {
            "dataframe": pd.get_dummies(
                data=dataframe,
                columns=self._columns_to_encode,
                drop_first=self._drop_first,
                dtype=int,
            ),
        }

        if verbose:
            logger.info(
                "Finished one-hot encoding. "
                f"Dataframe shape: {result['dataframe'].shape}"
            )

        return result