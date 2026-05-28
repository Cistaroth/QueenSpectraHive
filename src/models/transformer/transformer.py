import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from pipeline import ModelPipeline
from modules.utils.header import HeaderModule
from modules.data_loading.kaggle_loader import KaggleDataLoaderModule
from modules.data_loading.tabular_data_loader import TabularDataLoaderModule
from modules.tabular.tabular_utils import TabularColumnDropperModule
from modules.tabular.tabular_splitters import (
    TabularFeatureTargetSplitterModule,
    TabularTrainTestSplitterModule,
)
from modules.hyperparameter_tuning.hyperparameter_tuning import (
    HyperparameterTuningStratifiedKFoldModule,
)
from modules.evaluation.evaluation import ModelEvaluatorModule
from modules.data_augmentation.audio_splicing import AudioSplicerModule

from models.transformer.settings import AudioTransformerSettings

settings = AudioTransformerSettings()


def main() -> None:
    """
    Main function to run the Audio Transformer pipeline.
    """
    ModelPipeline(
        steps=[
            
            HeaderModule(task=settings.TASK_NAME),

            KaggleDataLoaderModule(
                dataset_handle=settings.DATASET_HANDLE,
                output_dir=settings.OUTPUT_DIR,
            ),

            TabularDataLoaderModule(
                filepath=settings.CSV_FILEPATH,
                max_samples=settings.MAX_SAMPLES,
            ),

            TabularColumnDropperModule(
                drop_columns=settings.DROP_COLUMNS,
            ),

            TabularFeatureTargetSplitterModule(
                target_column=settings.TARGET_COLUMN,
            ),

            TabularTrainTestSplitterModule(
                train_test_split=settings.TRAIN_TEST_SPLIT,
                random_state=settings.SEED,
            ),

            AudioSplicerModule(
                audio_path_col=settings.AUDIO_PATH_COL,
                audio_dir=settings.AUDIO_DIR,
                chunk_duration=settings.CHUNK_DURATION,
            ),
 
            HyperparameterTuningStratifiedKFoldModule(
                model_configuration=settings.HYPERPARAMETER_SETTINGS,
            ),
 
            ModelEvaluatorModule(
                inferencer=settings.HYPERPARAMETER_SETTINGS.model_inference,
                class_names=settings.class_names,
            ).set_dependency(["x_test", "y_test"], -3),
        ]
    ).run()


if __name__ == "__main__":
    main()