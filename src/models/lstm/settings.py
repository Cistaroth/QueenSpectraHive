from pathlib import Path

import torch.nn as nn
from pydantic import BaseModel

from config import config
from modules.hyperparameter_tuning.configuration import FineTuningConfiguration
from modules.lstm.lstm_train import FusionLSTMTrainModule
from modules.lstm.lstm_inference import FusionLSTMInferenceModule


class FusionLSTMSettings(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    TASK_NAME: str = "LSTM"
    SEED: int = config.SEED

    DATASET_HANDLE: str = "annajyang/beehive-sounds"
    OUTPUT_DIR: Path = Path(__file__).parents[2] / "data"
    CSV_FILEPATH: Path = OUTPUT_DIR / "all_data_updated.csv"
    SOUND_FILEPATH: Path = OUTPUT_DIR / "sound_files" / "sound_files"
    AUDIO_DIR: Path = OUTPUT_DIR / "sound_files" / "sound_files"
    PLOTS_DIR: Path = Path(__file__).parents[2] / "plots"

    AUDIO_PATH_COL: str = "file name"
    TARGET_COLUMN: str = "queen presence"

    GROUP_COLUMN: str = "hive number"

    TIME_COLUMN: str = "date"
    TIME_FEATURES: list[str] = ["hour", "minute", "day", "day_of_week", "week_of_year"]
    ONEHOT_COLUMNS: list[str] = ["device", "hive number"]
    IMPUTE_COLUMNS: list[str] = ["wind speed", "weather temp"]

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

    TRAIN_TEST_SPLIT: float = config.TRAIN_TEST_SPLIT
    CHUNK_DURATION: int = 15

    HYPERPARAMETER_SETTINGS: FineTuningConfiguration = FineTuningConfiguration(
        model_name="Fusion LSTM (audio + tabular)",
        model_train=FusionLSTMTrainModule,
        model_inference=FusionLSTMInferenceModule,
        model_hyperparameters={
            "drop_column":                [["file name", "start_sec", "end_sec"]],
            "embeddings_model":           [(nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 32))],
            "lstm_layers":                [(40, 128, 2, 0.3)],
            "classification_hidden_size": [64],
            "audio_dir":                  [str(AUDIO_DIR)],
            "n_mfcc":                     [40],
            "epochs":                     [7],
            "batch_size":                 [2],
            "learning_rate":              [1e-4],
        },
        metric="accuracy",
    )

    class_names: list[str] = ["Queen Absent", "Queen Present"]