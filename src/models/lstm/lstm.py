import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parents[2]))

from config import config
from models.lstm.settings import (
    AUDIO_DIR,
    AUDIO_PATH_COL,
    CHUNK_DURATION,
    CLASSIFICATION_HIDDEN_SIZE,
    CSV_FILEPATH,
    DATASET_HANDLE,
    DROP_COLUMNS,
    IMPUTE_COLUMNS,
    LSTM_ARCHITECTURE,
    NN_ARCHITECTURE,
    ONEHOT_COLUMNS,
    OUTPUT_DIR,
    SEED,
    SOUND_FILEPATH,
    TARGET_COLUMN,
    TASK_NAME,
    TIME_COLUMN,
    TIME_FEATURES,
    TRAIN_TEST_SPLIT,
)
from modules.data_augmentation.audio_splicing import AudioSplicerModule
from modules.data_loading.kaggle_loader import KaggleDataLoaderModule
from modules.data_loading.tabular_data_loader import TabularDataLoaderModule
#from modules.hyperparameter_tuning.hyperparameter_tuning import HyperparameterTuningStratifiedKFoldModule
from modules.lstm.composite_model_module import CompositeModelModule
from modules.tabular.tabular_feature_extractor import (
    TabularTimeColumnEncoderModule,
)
from modules.tabular.tabular_imputation import TabularColumnMeanImputerModule
from modules.tabular.tabular_splitters import (
    TabularFeatureTargetSplitterModule,
    TabularTrainTestSplitterModule,
)
from modules.tabular.tabular_utils import (
    TabularColumnDropperModule,
    TabularOneHotEncoderModule,
)
from modules.utils.header import HeaderModule
from pipeline import ModelPipeline

"""
So the main idea for the LSTM is to firstly, run the Mel spectrograms through the LSTM to get a audio summary vector
Then, we combine these with the tabular data ran through a Feed forwards neural network. You can't really pass tabular
data alongside audio data in a LSTM because the tabular data does not change with time but the audio does so it will get confused.

Finally, we concatonate both of these to give us a final prediction. To be fair it is kind of like an ensamble, anyways.
"""


def main():
    ModelPipeline(
        steps=[
            HeaderModule(task=TASK_NAME),
            KaggleDataLoaderModule(
                dataset_handle=DATASET_HANDLE,
                output_dir=OUTPUT_DIR,
            ),
            # Load tabular data
            TabularDataLoaderModule(filepath=CSV_FILEPATH),
            # Load audio data
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
            # Split features and target
            TabularFeatureTargetSplitterModule(target_column=TARGET_COLUMN),
            # Train-test split
            TabularTrainTestSplitterModule(
                train_test_split=TRAIN_TEST_SPLIT,
                random_state=SEED,
            ),
            # Get the audio files and stuff
            AudioSplicerModule(
                audio_path_col=AUDIO_PATH_COL,
                audio_dir=AUDIO_DIR,
                chunk_duration=CHUNK_DURATION,
            ),
            CompositeModelModule(
                ["file name", "start_sec", "end_sec"], NN_ARCHITECTURE, LSTM_ARCHITECTURE, CLASSIFICATION_HIDDEN_SIZE, SOUND_FILEPATH, 40, 5, 2, 1e-4, 0, config.SEED
            ),

        ],
    ).run()


if __name__ == "__main__":
    main()