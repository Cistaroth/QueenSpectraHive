import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pipeline import ModelPipeline
from modules.utils.header import HeaderModule
from modules.data_loading.kaggle_loader import KaggleDataLoaderModule
from modules.data_manipulation.data_loaders import (
    TabularDataLoaderModule,
    AudioDataLoaderModule,
)
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

sys.path.append(str(Path(__file__).parent.parent))

TASK_NAME = "LSTM"

DATASET_HANDLE = "annajyang/beehive-sounds"

OUTPUT_DIR = Path(__file__).parent.parent / "data"
CSV_FILEPATH = OUTPUT_DIR + "/all_data_updated.csv"
SOUND_FILEPATH = OUTPUT_DIR + "/sound_files/sound_files"


TIME_COLUMN = "date"
TIME_FEATURES = ["hour", "minute", "day", "day_of_week", "week_of_year"]

ONEHOT_COLUMNS = ["device", "hive number"]

IMPUTE_COLUMNS = ["wind speed", "weather temp"]

TARGET_COLUMN = "queen presence"


def main():
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

            KaggleDataLoaderModule(
            dataset_handle=DATASET_HANDLE,
            output_dir=OUTPUT_DIR,
            ),

            # Load tabular data
            TabularDataLoaderModule(filepath=CSV_FILEPATH),
            
            # Load audio data lazily
            AudioDataLoaderModule(filepath=SOUND_FILEPATH),
        ],
    ).run()


if __name__ == "__main__":
    main()



