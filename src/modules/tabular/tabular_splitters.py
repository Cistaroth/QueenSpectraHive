from typing import cast

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
    ) -> dict[str, pd.DataFrame | pd.Series]:
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

        result: dict[str, pd.DataFrame | pd.Series] = {
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

    def __init__(
        self,
        train_test_split: float = config.TRAIN_TEST_SPLIT,
        train_validation_split: float = config.TRAIN_VALIDATION_SPLIT,
        random_state: int = config.SEED,
    ) -> None:
        """
        Initialize the train-validation-test splitter class

        Args:
            train_test_split (float, optional): The proportion of the dataset to
                include in the test split. Defaults to config.TRAIN_TEST_SPLIT.
            train_validation_split (float, optional): The proportion of the
                training set to include in the validation split. Defaults to
                config.TRAIN_VALIDATION_SPLIT.
            random_state (int, optional): The random seed. Defaults to
                config.SEED.
        Returns:
            None
        """

        self._train_test_split = train_test_split
        self._train_validation_split = train_validation_split
        self._random_state = random_state

        super().__init__()

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
            console.section("Splitting dataframe into train, validation and test")

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self._train_test_split,
            random_state=self._random_state,
            stratify=target,
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=self._train_validation_split,
            random_state=self._random_state,
            stratify=y_train,
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

    def __init__(
        self,
        train_test_split: float = config.TRAIN_TEST_SPLIT,
        random_state: int = config.SEED,
        group_column: str | None = None,
    ) -> None:
        """
        Initialize the train-test splitter class

        Args:
            train_test_split (float, optional): The proportion of the dataset to
                include in the test split. Defaults to config.TRAIN_TEST_SPLIT.
            random_state (int, optional): The random seed. Defaults to
                config.SEED.
            group_column (str | None, optional): If given and present in the features,
                split by group so that every row sharing a group value
                lands entirely in train or entirely in test. Falls back to a stratified
                random row split when None or the column is absent. Defaults to None.
        Returns:
            None
        """

        self._train_test_split = train_test_split
        self._random_state = random_state
        self._group_column = group_column

        super().__init__()

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

        use_groups = (
            self._group_column is not None and self._group_column in features.columns
        )

        if use_groups:
            groups = features[self._group_column]
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=self._train_test_split,
                random_state=self._random_state,
            )
            train_idx, test_idx = next(splitter.split(features, target, groups=groups))
            x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
            if verbose:
                logger.info(
                    f"Group-aware split on '{self._group_column}'. "
                    f"Train groups: {sorted(groups.iloc[train_idx].unique())} | "
                    f"Test groups: {sorted(groups.iloc[test_idx].unique())}"
                )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=self._train_test_split,
                random_state=self._random_state,
                stratify=target,
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
                f"Y_test shape: {y_test.shape}\n"
                f"Train class balance:\n{y_train.value_counts()}\n"
                f"Test class balance:\n{y_test.value_counts()}"
            )

        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }
