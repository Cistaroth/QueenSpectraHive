from pathlib import Path

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
