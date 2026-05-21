import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline import ModelPipelineStep
from logger import console, logger
from config import config

class FeatureTargetSplitterModule(ModelPipelineStep):
    name = "FeatureTargetSplitter"
    inputs = {"dataframe"}
    outputs = {"features", "target"}

    def __init__(
        self,
        target_column: str = "queen presence",
    ) -> None:
        self._target_column = target_column

        super().__init__()

    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Split the dataframe into features and target

        Args:
            dataframe (pd.DataFrame): The dataframe to split
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: features, target
        """
        if verbose:
            console.section("Splitting dataframe into features and target")
        
        result = {
            "features": dataframe.drop(columns=self._target_column),
            "target": dataframe[self._target_column],
        }

        if verbose:
            logger.info(
                f"Finished splitting dataframe into features and target. \
                \nFeatures shape: {result['features'].shape}, \
                \nTarget shape: {result['target'].shape}"
            )

        return result

class TrainValidationTestSplitterModule(ModelPipelineStep):
    name= "TrainValidationTestSplitter"
    inputs = {"features", "target"}
    outputs = {"x_train", "x_val", "x_test", "y_train", "y_val", "y_test"}

    def run(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        verbose: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Split the dataframe into train, validation and test

        Args:
            dataframe (pd.DataFrame): The dataframe to split
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: x_train, x_val, x_test, y_train, y_val, y_test
        """
        if verbose:
            console.section("Splitting dataframe into train, validation and test")

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=config.TRAIN_TEST_SPLIT,
            random_state=config.SEED,
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=config.TRAIN_VALIDATION_SPLIT,
            random_state=config.SEED,
        )

        if verbose:
            logger.info(
                f"Finished splitting dataframe into train, validation and test. \
                \nX_train shape: {x_train.shape}, \
                \nX_val shape: {x_val.shape}, \
                \nX_test shape: {x_test.shape}, \
                \nY_train shape: {y_train.shape}, \
                \nY_val shape: {y_val.shape}, \
                \nY_test shape: {y_test.shape}"
            )

        return {
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }
    

class TrainTestSplitterModule(ModelPipelineStep):
    name = "TrainTestSplitter"
    inputs = {"features", "target"}
    outputs = {"x_train", "x_test", "y_train", "y_test"}

    def run(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        verbose: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Split the dataframe into train and test

        Args:
            dataframe (pd.DataFrame): The dataframe to split
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: x_train, x_test, y_train, y_test
        """
        if verbose:
            console.section("Splitting dataframe into train and test")
        
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=config.TRAIN_TEST_SPLIT,
            random_state=config.SEED,
        )

        if verbose:
            logger.info(
                f"Finished splitting dataframe into train and test. \
                \nX_train shape: {x_train.shape}, \
                \nX_test shape: {x_test.shape}, \
                \nY_train shape: {y_train.shape}, \
                \nY_test shape: {y_test.shape}"
            )

        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test
        }