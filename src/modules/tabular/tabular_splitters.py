from typing import cast 

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline import ModelPipelineStep
from logger import console, logger
from config import config


class TabularFeatureTargetSplitterModule(ModelPipelineStep):
    name = "TabularFeatureTargetSplitter"
    inputs = {"dataframe"}
    outputs = {"features", "target"}

    def __init__(
        self,
        target_column: str = "queen presence",
    ) -> None:
        """
        Initialize the feature-target splitter class

        Args:
            target_column (str, optional): The column to use as the target.
                Defaults to "queen presence".
        Returns:
            None
        """
        self._target_column = target_column

        super().__init__()

    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True,
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

        result: dict[str, pd.DataFrame] = {
            "features": dataframe.drop(columns=self._target_column),
            "target": dataframe[self._target_column],
        }

        if verbose:
            logger.info(
                "Finished splitting dataframe into features and target.\n"
                f"Features shape: {result['features'].shape},\n"
                f"Target shape: {result['target'].shape}"
            )

        return result


class TabularTrainValidationTestSplitterModule(ModelPipelineStep):
    name = "TabularTrainValidationTestSplitter"
    inputs = {"features", "target"}
    outputs = {"x_train", "x_val", "x_test", "y_train", "y_val", "y_test"}

    def run(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Split the dataframe into train, validation and test

        Args:
            features (pd.DataFrame): The features dataframe
            target (pd.DataFrame): The target series/dataframe
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: x_train, x_val, x_test, y_train, y_val,
                y_test
        """
        if verbose:
            console.section(
                "Splitting dataframe into train, validation and test"
            )

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

        x_train = cast(pd.DataFrame, x_train)
        x_val = cast(pd.DataFrame, x_val)
        x_test = cast(pd.DataFrame, x_test)
        y_train = cast(pd.DataFrame, y_train)
        y_val = cast(pd.DataFrame, y_val)
        y_test = cast(pd.DataFrame, y_test)

        if verbose:
            logger.info(
                "Finished splitting dataframe into train, validation and "
                "test.\n"
                f"X_train shape: {x_train.shape},\n"
                f"X_val shape: {x_val.shape},\n"
                f"X_test shape: {x_test.shape},\n"
                f"Y_train shape: {y_train.shape},\n"
                f"Y_val shape: {y_val.shape},\n"
                f"Y_test shape: {y_test.shape}"
            )

        return {
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }


class TabularTrainTestSplitterModule(ModelPipelineStep):
    name = "TabularTrainTestSplitter"
    inputs = {"features", "target"}
    outputs = {"x_train", "x_test", "y_train", "y_test"}

    def run(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Split the dataframe into train and test

        Args:
            features (pd.DataFrame): The features dataframe
            target (pd.DataFrame): The target series/dataframe
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

        x_train = cast(pd.DataFrame, x_train)
        x_test = cast(pd.DataFrame, x_test)
        y_train = cast(pd.DataFrame, y_train)
        y_test = cast(pd.DataFrame, y_test)
        
        if verbose:
            logger.info(
                "Finished splitting dataframe into train and test.\n"
                f"X_train shape: {x_train.shape},\n"
                f"X_test shape: {x_test.shape},\n"
                f"Y_train shape: {y_train.shape},\n"
                f"Y_test shape: {y_test.shape}"
            )

        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }