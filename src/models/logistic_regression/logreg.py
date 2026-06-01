import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from pipeline import ModelPipeline
from modules.utils.header import HeaderModule
from modules.data_loading.kaggle_loader import KaggleDataLoaderModule
from modules.data_loading.tabular_data_loader import TabularDataLoaderModule
from modules.tabular.tabular_utils import (
    TabularColumnDropperModule,
    TabularTrainTestOneHotEncoderModule,
)
from modules.tabular.tabular_feature_extractor import (
    TabularTimeColumnEncoderModule,
)
from modules.tabular.tabular_imputation import TabularTrainTestMeanImputerModule
from modules.tabular.tabular_splitters import (
    TabularFeatureTargetSplitterModule,
    TabularTrainTestSplitterModule,
)
from modules.hyperparameter_tuning.hyperparameter_tuning import (
    HyperparameterTuningStratifiedKFoldModule
)
from modules.evaluation.evaluation import ModelEvaluatorModule

from models.logistic_regression.settings import (
    LogisticRegressionSettings,
)

settings = LogisticRegressionSettings()

def main() -> None:
    ModelPipeline(
        steps = [
            HeaderModule(task=settings.TASK_NAME),

            KaggleDataLoaderModule(
                dataset_handle=settings.DATASET_HANDLE,
                output_dir=settings.OUTPUT_DIR,
            ),

            TabularDataLoaderModule(
                filepath=settings.CSV_FILEPATH
            ),

            TabularColumnDropperModule(
                drop_columns=settings.DROP_COLUMNS
            ),

            TabularTimeColumnEncoderModule(
                time_column=settings.TIME_COLUMN,
                time_features=settings.TIME_FEATURES,
            ),
            
            TabularColumnDropperModule(
                drop_columns=settings.TIME_FEATURES
            ),

            TabularFeatureTargetSplitterModule(
                target_column=settings.TARGET_COLUMN
            ),

            TabularTrainTestSplitterModule(
                train_test_split=settings.TRAIN_TEST_SPLIT,
                random_state=settings.SEED,
                group_column=settings.GROUP_COLUMN,
            ),

            TabularTrainTestOneHotEncoderModule(
                columns_to_encode=settings.ONEHOT_COLUMNS
            ),

            TabularTrainTestMeanImputerModule(
                impute_columns=settings.IMPUTE_COLUMNS
            ),

            HyperparameterTuningStratifiedKFoldModule(
                model_configuration=settings.HYPERPARAMETER_SETTINGS
            ).set_dependency("y_train", -3),

            ModelEvaluatorModule(
                inferencer=settings.HYPERPARAMETER_SETTINGS.model_inference,
                class_names=settings.class_names,
            ).set_dependency({"x_test": -2, "y_test": -4}),
        ],
    ).run()



if __name__ == "__main__":
    main()