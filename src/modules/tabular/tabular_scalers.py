from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from logger import console, logger
from modules.model_bases import TransformBase


class TabularMinMaxScalerModule(TransformBase):
    name = "MinMaxScaler"
    inputs = {"x_train", "x_test"}
    outputs = {"x_train_scaled", "x_test_scaled"}

    def __init__(self) -> None:
        """
        Initialize the min-max scaler class

        Args:
            None
        Returns:
            None
        """
        super().__init__()

        self.scaler = MinMaxScaler()

    def run(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, verbose: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Scale the data

        Args:
            x_train (pd.DataFrame): The training data
            x_test (pd.DataFrame): The test data
        Returns:
            dict[str, pd.DataFrame]: The scaled data
        """

        if verbose:
            console.section(title="Min-Max Scaling data")

        x_train_scaled = pd.DataFrame(
            data=self.fit_transform(x_train),
            columns=x_train.columns,
            index=x_train.index,
        )

        x_test_scaled = pd.DataFrame(
            data=self.transform(x_test), columns=x_test.columns, index=x_test.index
        )

        result = {"x_train_scaled": x_train_scaled, "x_test_scaled": x_test_scaled}

        if verbose:
            logger.info("Finished min-max scaling data.")

        return result

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler to the data and transform it

        Args:
            x (pd.DataFrame): The data to fit and transform
        Returns:
            pd.DataFrame: The scaled data
        """
        return pd.DataFrame(
            self.scaler.fit_transform(x), columns=x.columns, index=x.index
        )

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler

        Args:
            x (pd.DataFrame): The data to transform
        Returns:
            pd.DataFrame: The scaled data
        """
        return pd.DataFrame(self.scaler.transform(x), columns=x.columns, index=x.index)


class TabularStandardScalerModule(TransformBase):
    name = "StandardScaler"
    inputs = {"x_train", "x_test"}
    outputs = {"x_train_scaled", "x_test_scaled"}

    def __init__(self) -> None:
        """
        Initialize the standard scaler class

        Args:
            None
        Returns:
            None
        """
        super().__init__()

        self.scaler = StandardScaler()

    def run(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, verbose: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Scale the data

        Args:
            x_train (pd.DataFrame): The training data
            x_test (pd.DataFrame): The test data
        Returns:
            dict[str, pd.DataFrame]: The scaled data
        """

        if verbose:
            console.section(title="Standard Scaling data")

        x_train_scaled = pd.DataFrame(
            data=self.fit_transform(x_train),
            columns=x_train.columns,
            index=x_train.index,
        )

        x_test_scaled = pd.DataFrame(
            data=self.scaler.transform(x_test),
            columns=x_test.columns,
            index=x_test.index,
        )

        result = {"x_train_scaled": x_train_scaled, "x_test_scaled": x_test_scaled}

        if verbose:
            logger.info("Finished standard scaling data.")

        return result

    def fit_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler to the data and transform it

        Args:
            x (pd.DataFrame): The data to fit and transform
        Returns:
            pd.DataFrame: The scaled data
        """
        return pd.DataFrame(self.scaler.fit_transform(x), columns=x.columns, index=x.index)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler

        Args:
            x (pd.DataFrame): The data to transform
        Returns:
            pd.DataFrame: The scaled data
        """
        return pd.DataFrame(self.scaler.transform(x), columns=x.columns, index=x.index)
