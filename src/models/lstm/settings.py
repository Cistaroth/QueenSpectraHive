from pathlib import Path

import torch.nn as nn
from pydantic import BaseModel

from config import config
from modules.hyperparameter_tuning.configuration import FineTuningConfiguration
from modules.lstm.lstm_train import FusionLSTMTrainModule
from modules.lstm.lstm_inference import FusionLSTMInferenceModule
from modules.tabular.tabular_scalers import TabularStandardScalerModule
from modules.data_augmentation.tabular_interpolation import TabularSMOTE


class FusionLSTMSettings(BaseModel):
    """
    All configuration for the Fusion LSTM pipeline.

    The model fuses an audio branch (MFCC sequences -> LSTM) with a tabular
    branch (engineered features -> MLP). The two representations are concatenated
    and passed through a small feed-forward head for the final prediction.
    """

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

    # Rows are not independent: queen presence is near-constant within a hive over long
    # time blocks, so a random split leaks near-duplicate recordings across train/test.
    # Split by hive instead, holding whole hives out, for an honest estimate.
    GROUP_COLUMN: str = "hive number"

    # Tabular feature engineering
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
        folds=2,
        model_name="Fusion LSTM (audio + tabular)",
        model_train=FusionLSTMTrainModule,
        model_inference=FusionLSTMInferenceModule,
        transformer=TabularStandardScalerModule,
        resampler=TabularSMOTE,
        resampler_parameters={
            "audio_dir": AUDIO_DIR,
            "chunk_duration": CHUNK_DURATION,
        },
        group_column=GROUP_COLUMN,
        transformer_hyperparameters={
            "columns_to_exclude": [["file name", "start_sec", "end_sec"]],
        },
        model_hyperparameters={
            "drop_column":                [["file name", "start_sec", "end_sec"]],
            "embeddings_model":                     [(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 32))],
            # Shrunk from hidden=128 and raised dropout to fight the overfitting seen
            # on this small, few-hive dataset (seq_input_size, hidden, layers, dropout).
            "lstm_layers":                [(40, 64, 2, 0.4)],
            "classification_hidden_size": [64],
            "audio_dir":                  [str(AUDIO_DIR)],
            "n_mfcc":                     [40],
            # Ceiling only; early stopping (patience=5) ends training near the best epoch.
            "epochs":                     [30],
            "batch_size":                 [8],
            "learning_rate":              [1e-4],
            "weight_decay":               [1e-4],
            "patience":                   [5],
        },
        metric="accuracy",
    )

    class_names: list[str] = ["Queen Absent", "Queen Present"]