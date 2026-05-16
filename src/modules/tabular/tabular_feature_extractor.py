import pandas as pd
import numpy as np

from pipeline import ModelPipelineStep
from logger import console, logger


class TabularTimeColumnEncoderModule(ModelPipelineStep):
    name = "TabularTimeColumnEncoder"
    inputs = {"dataframe"}
    outputs = {"dataframe"}

    def __init__(
        self,
        time_column: str,
        time_features: list[str],
    ) -> None:
        """
        Initialize the time column encoder class

        Args:
            time_column (str): The time column
        Returns:
            None
        """
        self._extractors = {
            "year": lambda dt: dt.dt.year,
            "month": lambda dt: dt.dt.month,
            "day": lambda dt: dt.dt.day,
            "hour": lambda dt: dt.dt.hour,
            "minute": lambda dt: dt.dt.minute,
            "second": lambda dt: dt.dt.second,
            "day_of_year": lambda dt: dt.dt.dayofyear,
            "day_of_week": lambda dt: dt.dt.dayofweek,
            "week_of_year": lambda dt: dt.dt.isocalendar().week.astype(int),
            "quarter": lambda dt: dt.dt.quarter,
        }
        self._cyclical = {
            "hour": 24,
            "minute": 60,
            "second": 60,
            "day": 31,
            "day_of_year": 365,
            "day_of_week": 7,
            "month": 12,
        }

        self._time_column = time_column

        for feature in time_features:
            if feature not in self._extractors:
                valid_features = ", ".join(self._extractors.keys())
                raise ValueError(
                    f"Feature {feature} is not supported. "
                    f"Supported features: {valid_features}."
                )

        self._features = time_features

        super().__init__()

    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Encode the time column

        Args:
            dataframe (pd.DataFrame): The dataframe to encode
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: The encoded dataframe
        """
        if verbose:
            console.section(title="Time column encoding")
            logger.info(f"Time column: {self._time_column}")

        time_column = pd.to_datetime(dataframe[self._time_column])
        df = dataframe.drop(columns=self._time_column)

        for feature in self._features:
            df[feature] = self._extractors[feature](time_column)

            if feature in self._cyclical:
                period = self._cyclical[feature]
                df[f"{feature}_sin"] = np.sin(2 * np.pi * df[feature] / period)
                df[f"{feature}_cos"] = np.cos(2 * np.pi * df[feature] / period)

        result = {
            "dataframe": df,
        }

        if verbose:
            logger.info(
                "Finished time column encoding. "
                f"Dataframe shape: {result['dataframe'].shape}"
            )

        return result