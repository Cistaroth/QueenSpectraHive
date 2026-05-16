from pathlib import Path

from pipeline import ModelPipeline
from modules.utils.header import HeaderModule
from modules.data_loading.kaggle_loader import KaggleDataLoaderModule

TASK_NAME = "DATA DOWNLOADER"
DATASET_HANDLE = "annajyang/beehive-sounds"
OUTPUT_DIR = Path(__file__).parent / "data"

def main() -> None:
    """
    Main function to run the data downloader pipeline

    Args:
        None

    Returns:
        None
    """
    pipeline = ModelPipeline(
        steps=[
            HeaderModule(task=TASK_NAME),
            KaggleDataLoaderModule(
                dataset_handle=DATASET_HANDLE,
                output_dir=OUTPUT_DIR,
            )
        ]
    )

    pipeline.run()

if __name__ == "__main__":
    main()