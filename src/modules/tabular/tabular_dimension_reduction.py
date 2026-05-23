import pandas as pd
from sklearn.decomposition import PCA

from pipeline import ModelPipelineStep
from logger import console, logger

class TabularPCA(ModelPipelineStep):
    name = "Tabular PCA"

    inputs = {"x_train", "x_test"}
    outputs = {"pca", "x_train", "x_test"}

    def __init__(
        self,
        categorical_columns: list = [],
        n_components: float = 0.95,
        random_state: int = 67,
    ) -> None:
        """
        Initializes the TabularPCA class.

        Args:
            categorical_columns (list, optional): List of categorical column names to exclude from PCA. Defaults to [].
            n_components (float, optional): Number of principal components to retain. Defaults to 0.95.
        Returns:
            None
        """
        super().__init__()

        self._categorical_columns = categorical_columns
        self._n_components = n_components
        self._random_state = random_state

    def run(
            self,
            x_train: pd.DataFrame,
            x_test: pd.DataFrame,
            verbose: bool = True
            ) -> dict[str, pd.DataFrame | PCA]:
        """
        Runs PCA dimensionality reduction on the training and test features.

        Args:
            x_train (pd.DataFrame): Training features.
            x_test (pd.DataFrame): Test features.
            verbose (bool, optional): Verbose logging mode. Defaults to True.
        Returns:
            dict[str, PCA | pd.DataFrame]: Dictionary containing PCA object and transformed train/test sets.
        """

        if verbose:
            console.section("Applying PCA for Dimensionality Reduction on Tabular Data")

        pca = PCA(n_components=self._n_components, random_state=self._random_state)
        
        x_train_numerical = x_train.drop(columns=self._categorical_columns)
        x_test_numerical = x_test.drop(columns=self._categorical_columns)
        
        x_train_categorical = x_train[self._categorical_columns].reset_index(drop=True)
        x_test_categorical = x_test[self._categorical_columns].reset_index(drop=True)

        pca.fit(x_train_numerical)

        x_train_pca = pd.DataFrame(pca.transform(x_train_numerical), index=x_train.index)
        x_test_pca = pd.DataFrame(pca.transform(x_test_numerical), index=x_test.index)
        
        x_train_pca = pd.concat([x_train_pca, x_train_categorical], axis=1)
        x_test_pca = pd.concat([x_test_pca, x_test_categorical], axis=1)

        if verbose:
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info(f"Number of PCA components retained: {pca.n_components_}")
            logger.info(f"Total variance explained by PCA components: {explained_variance:.2%}")
            logger.info(f"Finished applying PCA. Transformed training set shape: {x_train_pca.shape}, Transformed test set shape: {x_test_pca.shape}")
            
        return {
            "pca": pca,
            "x_train": x_train_pca,
            "x_test": x_test_pca
        }

