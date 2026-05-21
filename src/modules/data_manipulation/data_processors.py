import pandas as pd
import numpy as np

from pipeline import ModelPipelineStep
from logger import console, logger

class OneHotEncoderModule(ModelPipelineStep):
    name = "OneHotEncoder"
    inputs = {"dataframe"}
    outputs = {"dataframe"}

    def __init__(
        self,
        columns_to_encode: list[str],
        drop_first: bool = True
    ) -> None:
        """
        Initialize the one-hot encoder class
        
        Args:
            columns_to_encode (list[str]): The columns to encode
            drop_first (bool, optional): Whether to drop the first column. Defaults to True.
        Returns:
            None
        """
        self._columns_to_encode = columns_to_encode
        self._drop_first = drop_first

        super().__init__()

    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True
    ) -> dict[str, pd.DataFrame]:
        if verbose:
            console.section(title="One-hot encoding")
            logger.info(f"Columns to encode: {self._columns_to_encode}")

        result = {
            "dataframe": pd.get_dummies(
                data=dataframe,
                columns=self._columns_to_encode,
                drop_first=self._drop_first,
                dtype=int
                )
        }

        if verbose:
            logger.info(f"Finished one-hot encoding. Dataframe shape: {result['dataframe'].shape}")

        return result
    
class TimeColumnEncoderModule(ModelPipelineStep):
    name = "TimeColumnEncoder"
    inputs = {"dataframe"}
    outputs = {"dataframe"}

    def __init__(
        self,
        time_column: str,
        features: list[str],
    ) -> None:
        """
        Initialize the time column encoder class
        
        Args:
            time_column (str): The time column
        Returns:
            None
        """
        self._extractors = {
            'year':         lambda dt: dt.dt.year,
            'month':        lambda dt: dt.dt.month,
            'day':          lambda dt: dt.dt.day,
            'hour':         lambda dt: dt.dt.hour,
            'minute':       lambda dt: dt.dt.minute,
            'second':       lambda dt: dt.dt.second,
            'day_of_year':  lambda dt: dt.dt.dayofyear,
            'day_of_week':  lambda dt: dt.dt.dayofweek,
            'week_of_year': lambda dt: dt.dt.isocalendar().week.astype(int),
            'quarter':      lambda dt: dt.dt.quarter,
        }
        self._cyclical = {
            'hour':        24,
            'minute':      60,
            'second':      60,
            'day':         31,
            'day_of_year': 365,
            'day_of_week': 7,
            'month':       12,
        }

        self._time_column = time_column

        for feature in features:
            if feature not in self._extractors.keys():
                valid_features = ", ".join(self._extractors.keys())
                raise ValueError(f"Feature {feature} is not supported. Supported features: {valid_features}.")
            
        self._features = features

        super().__init__()

    def run(
        self,
        dataframe: pd.DataFrame,
        verbose: bool = True
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
            "dataframe": df
        }

        if verbose:
            logger.info(f"Finished time column encoding. Dataframe shape: {result['dataframe'].shape}")

        return result