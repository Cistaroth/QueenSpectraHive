import math
from datetime import datetime

import pandas as pd

from pipeline import ModelPipelineStep


class TabularFeatureBuilderModule(ModelPipelineStep):
    """
    Builds the full 21-feature tabular row expected by the LSTM fusion model
    from the raw form fields submitted by the user.

    Feature engineering applied:
    - Numeric fields: passed through as-is (default 0.0 when absent).
    - Date/time: decomposed into cyclical sin/cos features.
    - device: one-hot encoded as device_2 (1 if device == 2, else 0).
    - hive number: one-hot encoded as hive number_3/4/5.
    """

    name = "TabularFeatureBuilder"
    inputs = {"x_test", "tabular_data"}
    outputs = {"x_test"}

    # Ordered to match the trained model's expected column sequence.
    FEATURE_COLUMNS = [
        "hive temp", "hive humidity", "hive pressure",
        "weather temp", "weather humidity", "weather pressure",
        "wind speed", "cloud coverage", "frames",
        "hour_sin", "hour_cos",
        "minute_sin", "minute_cos",
        "day_sin", "day_cos",
        "day_of_week_sin", "day_of_week_cos",
        "device_2",
        "hive number_3", "hive number_4", "hive number_5",
    ]

    _NUMERIC_MAP = [
        ("hive_temp",        "hive temp"),
        ("hive_humidity",    "hive humidity"),
        ("hive_pressure",    "hive pressure"),
        ("weather_temp",     "weather temp"),
        ("weather_humidity", "weather humidity"),
        ("weather_pressure", "weather pressure"),
        ("wind_speed",       "wind speed"),
        ("cloud_coverage",   "cloud coverage"),
        ("frames",           "frames"),
    ]

    def run(self, x_test: pd.DataFrame, tabular_data: dict) -> dict:
        row: dict = {}

        for form_key, col_name in self._NUMERIC_MAP:
            val = tabular_data.get(form_key)
            try:
                row[col_name] = float(val) if val not in (None, "") else 0.0
            except (ValueError, TypeError):
                row[col_name] = 0.0

        date_str = tabular_data.get("date")
        hour = minute = day = dow = 0
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                hour, minute, day, dow = dt.hour, dt.minute, dt.day, dt.weekday()
            except (ValueError, TypeError):
                pass

        row["hour_sin"]        = math.sin(2 * math.pi * hour   / 24)
        row["hour_cos"]        = math.cos(2 * math.pi * hour   / 24)
        row["minute_sin"]      = math.sin(2 * math.pi * minute / 60)
        row["minute_cos"]      = math.cos(2 * math.pi * minute / 60)
        row["day_sin"]         = math.sin(2 * math.pi * day    / 31)
        row["day_cos"]         = math.cos(2 * math.pi * day    / 31)
        row["day_of_week_sin"] = math.sin(2 * math.pi * dow    / 7)
        row["day_of_week_cos"] = math.cos(2 * math.pi * dow    / 7)

        try:
            device = int(tabular_data.get("device") or 1)
        except (ValueError, TypeError):
            device = 1
        row["device_2"] = 1.0 if device == 2 else 0.0

        try:
            hive_num = int(tabular_data.get("hive_number") or 1)
        except (ValueError, TypeError):
            hive_num = 1
        row["hive number_3"] = 1.0 if hive_num == 3 else 0.0
        row["hive number_4"] = 1.0 if hive_num == 4 else 0.0
        row["hive number_5"] = 1.0 if hive_num == 5 else 0.0

        tabular_df = pd.DataFrame([row], columns=self.FEATURE_COLUMNS)
        merged = pd.concat([x_test.reset_index(drop=True), tabular_df], axis=1)

        return {"x_test": merged}
