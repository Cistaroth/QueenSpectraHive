import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from modules.hyperparameter_tuning.hyperparameter_tuning import ScalerBase
from logger import console, logger


class MinMaxScalerModule(ScalerBase):
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
        self.scaler = MinMaxScaler()

        super().__init__()

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

    def fit_transform(self, x):
        return self.scaler.fit_transform(x)

    def transform(self, x):
        return self.scaler.transform(x)


class StandardScalerModule(ScalerBase):
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
        self.scaler = StandardScaler()

        super().__init__()

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

    def fit_transform(self, x):
        return self.scaler.fit_transform(x)

    def transform(self, x):
        return self.scaler.transform(x)
