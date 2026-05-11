from pathlib import Path

import kagglehub

from logger import console, logger
from pipeline import ModelPipelineStep

class KaggleDataLoaderModule(ModelPipelineStep):
    """
    Class for loading data from Kaggle
    """
    name = "KaggleDataLoader"
    inputs = {}
    outputs = {}
    
    def __init__(
        self,
        dataset_handle: str = "annajyang/beehive-sounds",
        output_dir: Path = Path(__file__).parent.parent / "data",
    ) -> None:
        """
        Initializes the KaggleDataLoader class

        Args:
            dataset_handle (str, optional): Kaggle dataset handle. Defaults to "annajyang/beehive-sounds".
            output_dir (Path, optional): Output directory. Defaults to Path(__file__).parent / "data".
        Returns:
            None
        """
        self._dataset_handle = dataset_handle
        self._output_dir = output_dir

        super().__init__()

    def run(
        self,
        verbose: bool = True
    )-> None:
        """
        Loads the data from Kaggle

        Args:
            verbose (bool, optional): Verbose mode. Defaults to True.
        Returns:
            None
        """
        if verbose:
            console.section(title="Loading data from Kaggle")
            logger.info(f"Dataset-Handle: [magenta]{self._dataset_handle}[/magenta]")
            logger.info(f"Loading into: {self._output_dir}")

        output_path = Path(self._output_dir)
        if output_path.exists() and any(output_path.iterdir()):
            if verbose:
                logger.info(f"Data already present in {self._output_dir}, skipping download.")
                
            return
        
        kagglehub.dataset_download(
            handle= self._dataset_handle,
            output_dir= self._output_dir,
        )

        if verbose:
            logger.info(f"Data successfully downloaded to {self._output_dir}")