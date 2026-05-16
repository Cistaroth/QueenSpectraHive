import pandas as pd

from pipeline import ModelPipelineStep
from logger import console, logger


class TabularColumnMeanImputerModule(ModelPipelineStep):
    name = "TabularColumnMeanImputer"
    inputs = {"dataframe"}
    outputs = {"dataframe"}

    def __init__(
        self,
        impute_columns: list[str] | None = None,
    ) -> None:
        """
        Initialize the tabular column mean imputer class

        Args:
            columns_to_impute (list[str], optional): The columns to impute.
                Defaults to None, which imputes all columns.
        Returns:
            None
        """
        self._impute_columns = impute_columns

        super().__init__()

    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Mean impute missing values in the dataframe

        Args:
            dataframe (pd.DataFrame): The dataframe to mean impute missing
                values in
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: The dataframe with missing values imputed
        """
        if verbose:
            console.section(title="Imputing missing values")
            logger.info(f"Columns to impute: {self._impute_columns}")

        if self._impute_columns is None:
            df = dataframe.fillna(dataframe.mean())
        else:
            df = dataframe.copy()
            df[self._impute_columns] = (
                df[self._impute_columns].fillna(
                    dataframe[self._impute_columns].mean()
            ))
            #df[self._impute_columns] = (
            #    df[self._impute_columns].fillna(dataframe.mean())
            #)

        result = {
            "dataframe": df,
        }

        if verbose:
            n_remaining = result["dataframe"].isna().sum().sum()
            logger.info(
                "Finished imputing missing values. "
                f"Number of remaining NAs: {n_remaining}"
            )

        return result