import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pipeline import ModelPipeline
from modules.utils.header import HeaderModule
from modules.data_loading.kaggle_loader import KaggleDataLoaderModule
from modules.tabular.tabular_data_loader import TabularDataLoaderModule
from modules.tabular.tabular_utils import (
    TabularColumnDropperModule,
    TabularOneHotEncoderModule,
)
from modules.tabular.tabular_feature_extractor import (
    TabularTimeColumnEncoderModule,
)
from modules.tabular.tabular_imputation import TabularColumnMeanImputerModule
from modules.tabular.tabular_splitters import (
    TabularFeatureTargetSplitterModule,
    TabularTrainTestSplitterModule,
)


TASK_NAME = "BASELINE LOGISTIC REGRESSION"

DATASET_HANDLE = "annajyang/beehive-sounds"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
CSV_FILEPATH = OUTPUT_DIR / "all_data_updated.csv"

DROP_COLUMNS = [
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

TIME_COLUMN = "date"
TIME_FEATURES = ["hour", "minute", "day", "day_of_week", "week_of_year"]

ONEHOT_COLUMNS = ["device", "hive number"]

IMPUTE_COLUMNS = ["wind speed", "weather temp"]

TARGET_COLUMN = "queen presence"


def main() -> None:
    """
    Main function to run the data downloader pipeline

    Args:
        None

    Returns:
        None
    """
    ModelPipeline(
        steps=[
            # Print header
            HeaderModule(task=TASK_NAME),

            # Load data from Kaggle
            KaggleDataLoaderModule(
                dataset_handle=DATASET_HANDLE,
                output_dir=OUTPUT_DIR,
            ),

            # Load tabular data
            TabularDataLoaderModule(filepath=CSV_FILEPATH),

            # Drop irrelevant columns
            TabularColumnDropperModule(drop_columns=DROP_COLUMNS),

            # Extract time features
            TabularTimeColumnEncoderModule(
                time_column=TIME_COLUMN,
                time_features=TIME_FEATURES,
            ),
            TabularColumnDropperModule(drop_columns=TIME_FEATURES),

            # One-hot encode
            TabularOneHotEncoderModule(columns_to_encode=ONEHOT_COLUMNS),

            # Mean impute missing values
            TabularColumnMeanImputerModule(impute_columns=IMPUTE_COLUMNS),

            # Split into features and target
            TabularFeatureTargetSplitterModule(target_column=TARGET_COLUMN),

            # Split into train and test
            TabularTrainTestSplitterModule(),
        ],
    ).run()


if __name__ == "__main__":
    main()