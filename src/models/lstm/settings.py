from pathlib import Path

import torch.nn as nn

from config import config
from modules.lstm.lstm_inference_module import LstmInferenceModule


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

NN_ARCHITECTURE = (nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 32))  # 14 features INCLUDING Y_train

LSTM_ARCHITECTURE = (40, 128, 2, 0.3)  # Seq_input_size, lstm_hidden_size, lstm_num_layers, lstm_dropout

CLASSIFICATION_HIDDEN_SIZE = 64

DROP_COLUMNS: list[str] = [
    "weatherID",
    "lat",
    "long",
    "rain",
    "queen acceptance",
    "target",
    "queen status",
    "time",
    "gust speed",
]
# device,hive number,date,hive temp,hive humidity,hive pressure,weather temp,weather humidity,weather pressure,wind speed,gust speed,cloud coverage,queen presence,frames
TIME_COLUMN: str = "date"

TIME_FEATURES: list[str] = ["hour", "minute", "day", "day_of_week", "week_of_year"]

TARGET_COLUMN: str = "queen presence"
ONEHOT_COLUMNS: list[str] = ["device", "hive number"]
IMPUTE_COLUMNS: list[str] = ["wind speed", "weather temp"]
TARGET_COLUMN: str = "queen presence"
SEED: int = config.SEED
TRAIN_TEST_SPLIT: float = config.TRAIN_TEST_SPLIT

AUDIO_PATH_COL: str = "file name"
AUDIO_DIR: Path = OUTPUT_DIR / "sound_files" / "sound_files"
CHUNK_DURATION: int = 15  # 15 seconds per slice

MODEL_INFERENCE = LstmInferenceModule

"""
HYPERPARAMETER_SETTINGS: FineTuningConfiguration = FineTuningConfiguration(
    model_name="LSTM-pre fusion",
    model_train=CompositeModelModule,
    model_inference=TransformerInferenceModule,
    transformer=AudioPassthroughTransformer,  # Scaler, not actual transformer
    model_hyperparameters={
        "audio_dir": [str(AUDIO_DIR)],
        "audio_path_col": ["file name"],
        "num_classes": [2],
        "epochs": [7],  #! short epochs for testing
        "batch_size": [2],
        "learning_rate": [1e-4],
    },
    metric="accuracy",
)
"""
