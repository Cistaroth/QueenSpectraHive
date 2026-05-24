from typing import Any
import pandas as pd

from modules.model_bases import TransformBase
from modules.tabular.tabular_scalers import TabularStandardScalerModule
from modules.tabular.tabular_dimension_reduction import TabularPCAModule
from logger import console, logger

class TabularScalePCATransformModule(TransformBase):
    name = "ScalePCATransformer"
    inputs = {"x_train", "x_test"}
    outputs = {"x_train_scaled_pca", "x_test_scaled_pca"}

    def __init__(
        self,
        n_components: float = 0.95,
        random_state: int = 67,
        categorical_columns: list = []
    ) -> None:
        """
        Initializes the TabularScalePCATransformModule class.

        Args:
            n_components (float, optional): Number of principal components to retain. Defaults to 0
            random_state (int, optional): Random state for PCA. Defaults to 67.
            categorical_columns (list, optional): List of categorical column names to exclude from PCA. Defaults
                [].
        Returns:
            None
        """
        super().__init__()

        self.scaler = TabularStandardScalerModule()
        self.pca = TabularPCAModule(
            n_components= n_components,
            random_state=random_state,
            categorical_columns=categorical_columns
        )
    
    def run(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        verbose: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Runs scaling and PCA transformation on the training and test features.

        Args:
            x_train (pd.DataFrame): Training features.
            x_test (pd.DataFrame): Test features.
            verbose (bool, optional): Verbose logging mode. Defaults to True.
        Returns:
            dict[str, pd.DataFrame]: Dictionary containing
                - "x_train_scaled_pca": Scaled and PCA-transformed training features.
                - "x_test_scaled_pca": Scaled and PCA-transformed test features.
        """

        if verbose:
            console.section(title="Scaling and applying PCA for dimensionality reduction on tabular data")

        x_train_scaled_pca = pd.DataFrame(
            data=self.fit_transform(x_train),
            columns=[f"PC{i+1}" for i in range(self.pca.pca.n_components_)],
            index=x_train.index,
        )

        x_test_scaled_pca = pd.DataFrame(
            data=self.transform(x_test),
            columns=[f"PC{i+1}" for i in range(self.pca.pca.n_components_)],
            index=x_test.index,
        )

        result = {
            "x_train_scaled_pca": x_train_scaled_pca,
            "x_test_scaled_pca": x_test_scaled_pca,
        }

        if verbose:
            logger.info("Finished scaling and applying PCA for dimensionality reduction on tabular data.")
        
        return result


    def fit_transform(self, x) -> Any:
        """
        Fit and scale the data and apply PCA transformation.

        Args:
            x (pd.DataFrame): The data to transform.
        Returns:
            pd.DataFrame: The scaled and PCA-transformed data.
        """
        x_transformed = self.scaler.fit_transform(x)
        x_transformed = self.pca.fit_transform(x)
        
        return x_transformed
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the data and apply PCA transformation.

        Args:
            x (pd.DataFrame): The data to transform.
        Returns:
            pd.DataFrame: The scaled and PCA-transformed data.
        """
        x_transformed = self.scaler.transform(x)
        x_transformed = self.pca.transform(x)

        return x_transformed