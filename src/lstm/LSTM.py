import sys
from pathlib import Path
import matplotlib.pyplot as plt

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
SOUND_FILEPATH = OUTPUT_DIR / "sound_files" / "sound_files"


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

    history = pipeline.run()
    dataset = history[-1].output["dataset"]
    
    print("\nFETCHING DATA FROM LAZY AUDIO LOADER...")
    # Trigger __getitem__(0) to fetch a single sample
    waveform = dataset[0]
    
    print(f" Received Waveform shape: {waveform.shape}")

    extractor = MFCCExtractor(n_mfcc=40)
    extraction_results = extractor.run(
        waveform=waveform,
        sample_rate=16000,
        verbose=True
    )
    
    print("\nFINAL TEST VERIFICATION:")
    print(f"  Final MFCC Matrix Tensor Shape: {extraction_results['mfcc'].shape}")

    print("\n Generating visualization plots...")
    

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Audio Session", fontsize=14, fontweight='bold')

    waveform_np = waveform[0].numpy()
    ax1.plot(waveform_np, color='#1f77b4', alpha=0.7)
    ax1.set_title("Stitched & Spliced Waveform (Time Domain)")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.5)

    mfcc_np = extraction_results['mfcc'][0].numpy()
    im = ax2.imshow(mfcc_np, cmap='viridis', origin='lower', aspect='auto')
    ax2.set_title("Extracted MFCC Coefficients (Frequency Domain Feature Map)")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("MFCC Coefficients")
    fig.colorbar(im, ax=ax2, label="dB / Energy")

    plt.tight_layout()
    
    output_image_path = Path(__file__).parent.parent / "audio_test_verification.png"
    plt.savefig(output_image_path, dpi=150)
    print(f"Success! Plot saved cleanly to: {output_image_path}")


if __name__ == "__main__":
    main()
