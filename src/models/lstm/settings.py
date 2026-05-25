from pathlib import Path
import torch.nn as nn


TASK_NAME = "LSTM"

DATASET_HANDLE = "annajyang/beehive-sounds"

OUTPUT_DIR = Path(__file__).parents[2] / "data"
CSV_FILEPATH = OUTPUT_DIR / "all_data_updated.csv"
SOUND_FILEPATH = OUTPUT_DIR / "sound_files" / "sound_files"


TIME_COLUMN = "date"
TIME_FEATURES = ["hour", "minute", "day", "day_of_week", "week_of_year"]

ONEHOT_COLUMNS = ["device", "hive number"]

IMPUTE_COLUMNS = ["wind speed", "weather temp"]

TARGET_COLUMN = "queen presence"

NN_ARCHITECTURE = nn.Linear(14, 64),nn.ReLU(),nn.Linear(64, 32) #14 features INCLUDING Y_train

DROP_COLUMNS: list[str] = [
    "weatherID",
    "lat",
    "long",
    "rain",
    "file name",
    "queen acceptance",
    "target",
    "queen status",
    "time",
    "gust speed",
]
# device,hive number,date,hive temp,hive humidity,hive pressure,weather temp,weather humidity,weather pressure,wind speed,gust speed,cloud coverage,queen presence,frames
TIME_COLUMN: str = "date"

TIME_FEATURES: list[str] = [
    "hour",
    "minute",
    "day",
    "day_of_week",
    "week_of_year"
]
ONEHOT_COLUMNS: list[str] = ["device", "hive number"]
IMPUTE_COLUMNS: list[str] = ["wind speed", "weather temp"]
TARGET_COLUMN: str = "queen presence"