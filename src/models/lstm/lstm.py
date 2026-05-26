import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parents[2]))

from models.lstm.settings import (
    CSV_FILEPATH,
    DATASET_HANDLE,
    OUTPUT_DIR,
    SOUND_FILEPATH,
    TASK_NAME,
    NN_ARCHITECTURE,
    DROP_COLUMNS,
    TIME_COLUMN,
    TIME_FEATURES,
    ONEHOT_COLUMNS,
    IMPUTE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_TEST_SPLIT,
    SEED,
    AUDIO_DIR,
    AUDIO_PATH_COL,
    CHUNK_DURATION
)

from modules.data_loading.kaggle_loader import KaggleDataLoaderModule
from modules.data_loading.tabular_data_loader import TabularDataLoaderModule
from modules.feature_extraction.mfcc_extraction import MFCCExtractorModule
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

from modules.data_augmentation.audio_splicing import AudioSplicerModule
from modules.lstm.tabularnn_embedding import TabularNNEmbeddingsModule



'''
So the main idea for the LSTM is to firstly, run the Mel spectrograms through the LSTM to get a audio summary vector
Then, we combine these with the tabular data ran through a Feed forwards neural network. You can't really pass tabular
data alongside audio data in a LSTM because the tabular data does not change with time but the audio does so it will get confused.

Finally, we concatonate both of these to give us a final prediction. To be fair it is kind of like an ensamble, anyways.
'''


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

            # Obtain the NN embeddings
            TabularNNEmbeddingsModule(*NN_ARCHITECTURE),

            #Get the audio files and stuff
            AudioSplicerModule(
                audio_path_col=AUDIO_PATH_COL,
                audio_dir=AUDIO_DIR,
                chunk_duration=CHUNK_DURATION,
            ),
        ],
    ).run()

if __name__ == "__main__":
    main()


"""
    history = pipeline.run()
    dataset = history[-1].output["dataset"]

    print("\nFETCHING DATA FROM LAZY AUDIO LOADER...")
    # Trigger __getitem__(0) to fetch a single sample
    waveform = dataset[0]

    print(f" Received Waveform shape: {waveform.shape}")

    extractor = MFCCExtractorModule(n_mfcc=40)
    extraction_results = extractor.run(waveform=waveform, sample_rate=16000, verbose=True)

    print("\nFINAL TEST VERIFICATION:")
    print(f"  Final MFCC Matrix Tensor Shape: {extraction_results['mfcc'].shape}")

    print("\n Generating visualization plots...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f"Audio Session", fontsize=14, fontweight="bold")

    waveform_np = waveform[0].numpy()
    ax1.plot(waveform_np, color="#1f77b4", alpha=0.7)
    ax1.set_title("Stitched & Spliced Waveform (Time Domain)")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle="--", alpha=0.5)

    mfcc_np = extraction_results["mfcc"][0].numpy()
    im = ax2.imshow(mfcc_np, cmap="viridis", origin="lower", aspect="auto")
    ax2.set_title("Extracted MFCC Coefficients (Frequency Domain Feature Map)")
    ax2.set_xlabel("Time Frames")
    ax2.set_ylabel("MFCC Coefficients")
    fig.colorbar(im, ax=ax2, label="dB / Energy")

    plt.tight_layout()

    output_image_path = Path(__file__).parents[2] / "plots" / "audio_test_verification.png"
    plt.savefig(output_image_path, dpi=150)
    print(f"Success! Plot saved cleanly to: {output_image_path}")
"""