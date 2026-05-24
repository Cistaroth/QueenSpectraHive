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

from modules.data_augmentation.mfcc_extraction import MFCCExtractor

sys.path.append(str(Path(__file__).parent.parent))

TASK_NAME = "LSTM"

DATASET_HANDLE = "annajyang/beehive-sounds"

OUTPUT_DIR = Path(__file__).parent.parent / "data"
CSV_FILEPATH = OUTPUT_DIR / "all_data_updated.csv"
SOUND_FILEPATH = OUTPUT_DIR / "/sound_files" / "sound_files"


TIME_COLUMN = "date"
TIME_FEATURES = ["hour", "minute", "day", "day_of_week", "week_of_year"]

ONEHOT_COLUMNS = ["device", "hive number"]

IMPUTE_COLUMNS = ["wind speed", "weather temp"]

TARGET_COLUMN = "queen presence"


def main():
    pipeline = ModelPipeline(
        steps=[
            HeaderModule(task=TASK_NAME),

            KaggleDataLoaderModule(
                dataset_handle=DATASET_HANDLE,
                output_dir=OUTPUT_DIR,
            ),

            TabularDataLoaderModule(filepath=CSV_FILEPATH),
            AudioDataLoaderModule(filepath=SOUND_FILEPATH),
        ],
    )

    # 1. Run the base data setups
    history = pipeline.run()

    # 2. 🚀 THE ACID TEST: Grab the generated dataset from the pipeline history
    dataset = history[-1].output["dataset"]
    
    print("\n📬 FETCHING DATA FROM TEAMMATE'S LAZY AUDIO LOADER...")
    # Trigger __getitem__(0) to fetch a single sample
    waveform, start_sec, end_sec = dataset[0]
    
    print(f"   • Received Waveform shape: {waveform.shape}")
    print(f"   • Received Slicing Windows: {start_sec}s -> {end_sec}s")

    # 3. 🚀 Pass those exact extracted variables directly into your MFCCExtractor
    extractor = MFCCExtractor(n_mfcc=40)
    extraction_results = extractor.run(
        waveform=waveform,
        sample_rate=16000, # target_sample_rate specified in their loader
        start_sec=start_sec,
        end_sec=end_sec,
        verbose=True
    )
    
    print("\n🏁 FINAL TEST VERIFICATION:")
    print(f"   • Final MFCC Matrix Tensor Shape: {extraction_results['mfcc'].shape}")


if __name__ == "__main__":
    main()
