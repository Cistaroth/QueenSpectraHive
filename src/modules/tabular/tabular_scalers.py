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

    def __init__(self, columns_to_exclude: list[str] | None = None) -> None:
        """
        Initialize the standard scaler class

        Args:
            columns_to_exclude (list[str] | None): Columns to pass through without scaling
                (e.g. audio path or id columns). Defaults to None (scale all columns).
        Returns:
            None
        """
        super().__init__()

        self._columns_to_exclude = set(columns_to_exclude) if columns_to_exclude else set()
        self.scaler = StandardScaler()

    def _scale_cols(self, x: pd.DataFrame) -> list[str]:
        return [c for c in x.columns if c not in self._columns_to_exclude]

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

        x_train_scaled = self.fit_transform(x_train)
        x_test_scaled = self.transform(x_test)

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
        cols = self._scale_cols(x)
        scaled = pd.DataFrame(self.scaler.fit_transform(x[cols]), columns=cols, index=x.index)
        if self._columns_to_exclude:
            passthrough = x[[c for c in x.columns if c in self._columns_to_exclude]]
            return pd.concat([scaled, passthrough], axis=1)[x.columns]
        return scaled

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler

        Args:
            x (pd.DataFrame): The data to transform
        Returns:
            pd.DataFrame: The scaled data
        """
        cols = self._scale_cols(x)
        scaled = pd.DataFrame(self.scaler.transform(x[cols]), columns=cols, index=x.index)
        if self._columns_to_exclude:
            passthrough = x[[c for c in x.columns if c in self._columns_to_exclude]]
            return pd.concat([scaled, passthrough], axis=1)[x.columns]
        return scaled
