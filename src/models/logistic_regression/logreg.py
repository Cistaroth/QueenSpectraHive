import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2]))

from pipeline import ModelPipeline
from modules.utils.header import HeaderModule
#from modules.data_loading.kaggle_loader import KaggleDataLoaderModule
from modules.data_loading.tabular_data_loader import TabularDataLoaderModule
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
from modules.hyperparameter_tuning.hyperparameter_tuning import (
    HyperparameterTuningStratifiedKFoldModule
)
from modules.evaluation.evaluation import ModelEvaluatorModule

from models.logistic_regression.settings import (
    LogisticRegressionSettings,
)

settings = LogisticRegressionSettings()

def main() -> None:
    """
    Main function to run the data downloader pipeline

    Args:
        None

    Returns:
        None
    """
    ModelPipeline(
        steps = [
            # Print header
            HeaderModule(task=settings.TASK_NAME),

            # Load data from Kaggle
            # KaggleDataLoaderModule(
            #     dataset_handle=settings.DATASET_HANDLE,
            #     output_dir=settings.OUTPUT_DIR,
            # ),

            # Load tabular data
            TabularDataLoaderModule(
                filepath=settings.CSV_FILEPATH
            ),

            # Drop irrelevant columns
            TabularColumnDropperModule(
                drop_columns=settings.DROP_COLUMNS
            ),

            # Extract time features
            TabularTimeColumnEncoderModule(
                time_column=settings.TIME_COLUMN,
                time_features=settings.TIME_FEATURES,
            ),
            
            TabularColumnDropperModule(
                drop_columns=settings.TIME_FEATURES
            ),

            # One-hot encode
            TabularOneHotEncoderModule(
                columns_to_encode=settings.ONEHOT_COLUMNS
            ),

            # Mean impute missing values
            TabularColumnMeanImputerModule(
                impute_columns=settings.IMPUTE_COLUMNS
            ),

            # Split features and target
            TabularFeatureTargetSplitterModule(
                target_column=settings.TARGET_COLUMN
            ),

            # Train-test split
            TabularTrainTestSplitterModule(
                train_test_split=settings.TRAIN_TEST_SPLIT,
                random_state=settings.SEED,
            ),

            # Model Training using K-Fold Cross Validation
            HyperparameterTuningStratifiedKFoldModule(
                model_configuration=settings.HYPERPARAMETER_SETTINGS
            ),

            # Model Evaluation
            ModelEvaluatorModule(
                inferencer=settings.HYPERPARAMETER_SETTINGS.model_inference,
                class_names=settings.class_names,
            ).set_dependency(["x_test", "y_test"], -2),
        ],
    ).run()



if __name__ == "__main__":
    main()