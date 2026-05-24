from pathlib import Path

from pydantic import BaseModel

from modules.hyperparameter_tuning.hyperparameter_tuning import FineTuningConfiguration
from modules.logreg.logreg_train import LogRegTrainModule
from modules.logreg.logreg_inference import LogRegInferenceModule
from modules.tabular.tabular_scalers import TabularStandardScalerModule
from config import config

class LogisticRegressionSettings(BaseModel):
    TASK_NAME: str = "LOGISTIC REGRESSION"
    SEED: int = config.SEED
    DATASET_HANDLE: str = "annajyang/beehive-sounds"
    OUTPUT_DIR: Path = Path(__file__).parents[2] / "data"
    CSV_FILEPATH: Path = OUTPUT_DIR / "all_data_updated.csv"
    PLOTS_DIR: Path = Path(__file__).parents[2] / "plots"
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
    TRAIN_TEST_SPLIT: float = config.TRAIN_TEST_SPLIT

    HYPERPARAMETER_SETTINGS: FineTuningConfiguration = FineTuningConfiguration(
        model_name="Logistic Regression",
        model_train=LogRegTrainModule,
        model_inference=LogRegInferenceModule,
        transformer=TabularStandardScalerModule,
        model_hyperparameters={
            "C": [0.01, 0.1, 1, 10, 100],
            "l1_ratio": [0, 0.25, 0.5, 0.75, 1],
            "solver": ["saga"],
            "class_weight": ["balanced"],
            "max_iter": [10000],
        },
        metric="accuracy",
    )
    class_names: list[str] = ["Queen Absent", "Queen Present"]
