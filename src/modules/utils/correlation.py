from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pipeline import ModelPipelineStep
from logger import console, logger

class TabularCorrelation(ModelPipelineStep):
    name = "Correlation Analysis"

    inputs = {"dataframe"}
    outputs = {"correlation_matrix"}

    def __init__(self, save_path: Path) -> None:
        """
        Initializes the Correlation class.

        Args:
            None
        Returns:
            None
        """
        super().__init__()

        self._save_path = save_path

    def run(
            self,
            dataframe: pd.DataFrame,
            verbose: bool = True
            ) -> dict[str, pd.DataFrame | pd.Series]:
        """
        Runs correlation analysis on the training features.

        Args:
            x_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target labels.
            verbose (bool, optional): Verbose logging mode. Defaults to True.
        Returns:
            dict[str, Any]: Dictionary containing balanced train sets and untouched test sets.
        """

        if verbose:
            console.section("Correlation Analysis of Tabular Features")

        correlation_matrix = dataframe.corr()

        if verbose:
            plt.figure(figsize=(22, 8))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt=".1f",
                annot_kws={"size": 8},
                cmap="viridis",
                linewidths=0.5
            )
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.tight_layout()
            plt.title("Correlation Matrix of Tabular Features", fontsize=16)
            plt.savefig(self._save_path, dpi=150, bbox_inches='tight')

            logger.info(f"Correlation matrix saved to: {self._save_path}")


        return {
            "correlation_matrix": correlation_matrix
        }