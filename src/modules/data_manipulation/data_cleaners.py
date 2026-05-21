import pandas as pd

from pipeline import ModelPipelineStep
from logger import console, logger

class TabularColumnDropperModule(ModelPipelineStep):
    name = "TabularColumnDropper"
    inputs = {"dataframe"}
    outputs = {"dataframe"}

    def __init__(
        self,
        drop_columns: list[str]
    ) -> None:
        self._drop_columns = drop_columns
        
        super().__init__()
    
    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True
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
            "dataframe": dataframe.drop(columns=self._drop_columns)
        }

        if verbose:
            logger.info(f"Finished dropping columns. Dataframe shape: {result['dataframe'].shape}")

        return result
    
class TabularColumnMeanImputerModule(ModelPipelineStep):
    name = "TabularColumnMeanImputer"
    inputs = {"dataframe"}
    outputs = {"dataframe"}

    def __init__(
        self,
        columns_to_impute: list[str] = []
    ) -> None:
        """
        Initialize the tabular column mean imputer class

        Args:
            columns_to_impute (list[str], optional): The columns to impute. 
                Defaults to [], which imputes all columns.
        Returns:
            None
        """
        self._columns_to_impute = columns_to_impute

        super().__init__()

    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Mean impute missing values in the dataframe

        Args:
            dataframe (pd.DataFrame): The dataframe to mean impute missing values in
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: The dataframe with missing values imputed
        """
        if verbose:
            console.section(title="Imputing missing values")
            logger.info(f"Columns to impute: {self._columns_to_impute}")

        if self._columns_to_impute is None:
            df = dataframe.fillna(dataframe.mean())
        else:
            df = dataframe.copy()
            df[self._columns_to_impute] = df[self._columns_to_impute].fillna(dataframe.mean())

        result = {
            "dataframe": df
        }

        if verbose:
            logger.info(f"Finished imputing missing values. Number of remaining NAs: {result['dataframe'].isna().sum().sum()}")

        return result