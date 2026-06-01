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
            df[self._impute_columns] = df[self._impute_columns].fillna(
                dataframe[self._impute_columns].mean()
            )

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


class TabularTrainTestMeanImputerModule(ModelPipelineStep):
    name = "TabularTrainTestMeanImputer"
    inputs = {"x_train", "x_test"}
    outputs = {"x_train", "x_test"}

    def __init__(
        self,
        impute_columns: list[str] | None = None,
    ) -> None:
        self._impute_columns = impute_columns
        super().__init__()

    def run(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        if verbose:
            console.section(title="Imputing missing values (train only)")
            logger.info(f"Columns to impute: {self._impute_columns}")

        cols = (
            self._impute_columns
            if self._impute_columns is not None
            else x_train.columns.tolist()
        )
        train_means = x_train[cols].mean()

        x_train_out = x_train.copy()
        x_train_out[cols] = x_train[cols].fillna(train_means)

        x_test_out = x_test.copy()
        x_test_out[cols] = x_test[cols].fillna(train_means)

        if verbose:
            n_train = x_train_out.isna().sum().sum()
            n_test = x_test_out.isna().sum().sum()
            logger.info(
                f"Finished imputing. Remaining NAs - train: {n_train}, test: {n_test}"
            )

        return {"x_train": x_train_out, "x_test": x_test_out}
