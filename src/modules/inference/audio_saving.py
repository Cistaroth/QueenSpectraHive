import math
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import soundfile as sf
from fastapi import UploadFile

from logger import console, logger
from pipeline import ModelPipelineStep


class LSTMAudioSavingModule(ModelPipelineStep):
    name = "LSTMAudioSavingModule"
    inputs = {"file", "tabular_data"}
    outputs = {"result", "x_test"}

    FEATURE_COLUMNS = [
        "hive temp",
        "hive humidity",
        "hive pressure",
        "frames",
        "weather temp",
        "weather humidity",
        "weather pressure",
        "wind speed",
        "cloud coverage",
        "hour_sin",
        "hour_cos",
        "minute_sin",
        "minute_cos",
        "day_sin",
        "day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "device_2",
        "hive number_3",
        "hive number_4",
        "hive number_5",
    ]

    _NUMERIC_COLUMNS = [
        ("hive_temp", "hive temp"),
        ("hive_humidity", "hive humidity"),
        ("hive_pressure", "hive pressure"),
        ("frames", "frames"),
        ("weather_temp", "weather temp"),
        ("weather_humidity", "weather humidity"),
        ("weather_pressure", "weather pressure"),
        ("wind_speed", "wind speed"),
        ("cloud_coverage", "cloud coverage"),
    ]

    def __init__(
        self,
        save_path: Path = Path(__file__).parents[2]
        / "data"
        / "sound_files"
        / "sound_files",
        chunk_duration: float = 360.0,
    ) -> None:

        super().__init__()
        self._save_path = save_path
        self._chunk_duration = float(chunk_duration)

    def run(
        self,
        file: UploadFile,
        tabular_data: dict,
        verbose: bool = False,
    ) -> dict[str, pd.DataFrame]:
        if verbose:
            console.section(title="Saving audio file")
            console.print(f"Saving audio file to {self._save_path}")

        save_path = self._save_path / "inference_input__segment.wav"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        file.file.seek(0)
        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        if verbose:
            console.print(f"Audio file saved to {save_path}")

        try:
            sound_info = sf.info(save_path)
            duration_sec = sound_info.frames / sound_info.samplerate
        except Exception as e:
            logger.warning(
                f"Could not read duration of saved audio at {save_path}: {e}. "
                "Falling back to a full-file crop."
            )
            duration_sec = self._chunk_duration

        if duration_sec <= self._chunk_duration:
            start_sec = 0.0
            end_sec = float(duration_sec)
        else:
            start_sec = (duration_sec - self._chunk_duration) / 2.0
            end_sec = start_sec + self._chunk_duration

        if verbose:
            logger.info(
                f"Audio duration: {duration_sec:.2f}s; "
                f"centered crop -> start_sec={start_sec:.2f}s, end_sec={end_sec:.2f}s "
                f"(chunk_duration={self._chunk_duration:.2f}s)"
            )

        audio_meta = pd.DataFrame(
            {
                "file name": ["inference_input.raw"],
                "start_sec": [start_sec],
                "end_sec": [end_sec],
            }
        )

        tabular_df = pd.DataFrame(
            [self._build_tabular_row(tabular_data)],
            columns=self.FEATURE_COLUMNS,
        )

        x_test = pd.concat([audio_meta, tabular_df], axis=1)

        return {"x_test": x_test}

    def _build_tabular_row(self, tabular_data: dict) -> dict:
        """
        Converts raw tabular input data into a feature row for the model, including numeric features and cyclical encodings for time-based features.

        Args:
            tabular_data (dict): The raw input data containing keys like "hive_temp",
                "hive_humidity", "hive_pressure", "frames", "weather_temp",
                "weather_humidity", "weather_pressure", "wind_speed",
                "cloud_coverage", and "date".
        Returns:
            dict: A dictionary representing a single row of features for the model, with numeric values and
        """
        row = {}

        for form_key, col_name in self._NUMERIC_COLUMNS:
            val = tabular_data.get(form_key)
            try:
                row[col_name] = float(val) if val not in (None, "") else 0.0
            except (ValueError, TypeError):
                row[col_name] = 0.0

        hour, minute, day, dow = 0, 0, 0, 0
        date_str = tabular_data.get("date")
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                hour, minute, day, dow = dt.hour, dt.minute, dt.day, dt.weekday()
            except (ValueError, TypeError):
                pass

        row["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        row["hour_cos"] = math.cos(2 * math.pi * hour / 24)
        row["minute_sin"] = math.sin(2 * math.pi * minute / 60)
        row["minute_cos"] = math.cos(2 * math.pi * minute / 60)
        row["day_sin"] = math.sin(2 * math.pi * day / 31)
        row["day_cos"] = math.cos(2 * math.pi * day / 31)
        row["day_of_week_sin"] = math.sin(2 * math.pi * dow / 7)
        row["day_of_week_cos"] = math.cos(2 * math.pi * dow / 7)

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

        return row
