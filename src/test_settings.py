from pathlib import Path

from pydantic import BaseModel

from modules.hyperparameter_tuning.hyperparameter_tuning import FineTuningConfiguration
from modules.transformer.transformer_train import TransformerTrainModule
from modules.transformer.transformer_inference import TransformerInferenceModule
from config import config


class AudioTransformerSettings(BaseModel):
    """
    All configuration for the Audio Transformer pipeline.

    Only the audio file path and thetarget label are saved
    .Every tabular feature is discarded so the model trains
    exclusively on the audio data.

    """

    model_config = {"arbitrary_types_allowed": True}

    TASK_NAME: str = "AUDIO TRANSFORMER"
    SEED: int = config.SEED

    DATASET_HANDLE: str = "annajyang/beehive-sounds"
    OUTPUT_DIR: Path = Path(__file__).parent / "data"
    CSV_FILEPATH: Path = OUTPUT_DIR / "all_data_updated.csv"
    AUDIO_DIR: Path = OUTPUT_DIR / "sound_files" / "sound_files"
    PLOTS_DIR: Path = Path(__file__).parent / "plots"

    AUDIO_PATH_COL: str = "file name"
    TARGET_COLUMN: str = "queen presence"

    # Drop every column except AUDIO_PATH_COL and TARGET_COLUMN.
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
        "date",
        "device",
        "hive number",
        "wind speed",
        "weather temp",
    ]

    TRAIN_TEST_SPLIT: float = config.TRAIN_TEST_SPLIT
    MAX_SAMPLES: int | None = None   #! set max samples to 10 for testing, change 10 to None for full dataset

    CHUNK_DURATION: int = 15           # seconds per slice

    HYPERPARAMETER_SETTINGS: FineTuningConfiguration = FineTuningConfiguration(
        model_name="Pretrained Audio Spectrogram Transformer",
        model_train=TransformerTrainModule,
        model_inference=TransformerInferenceModule,
        model_hyperparameters={
            "audio_dir":      [str(AUDIO_DIR)],
            "audio_path_col": ["file name"],
            "num_classes":    [1],
            "pretrained_model": ["MIT/ast-finetuned-audioset-10-10-0.4593"],
            "epochs":         [7],
            "batch_size":     [2],
            "learning_rate":  [1e-4],
        },
        metric="accuracy",
    )

    class_names: list[str] = ["Queen Absent", "Queen Present"]