from typing import Any

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from rich import box
from rich.table import Table
from rich.text import Text
from rich.console import Group
from rich.align import Align

from modules.model_bases import InferencerBase, TransformBase
from pipeline import ModelPipelineStep
from logger import console, logger


class ModelEvaluatorModule(ModelPipelineStep):
    name = "ModelEvaluator"
    inputs = {"model", "transformer", "x_test", "y_test"}
    outputs = {
        "x_test_scaled", "y_pred", "confusion_matrix",
        "accuracy", "roc_auc", "precision", "recall", "f1",
    }

    def __init__(
        self,
        inferencer: type[InferencerBase]
    ) -> None:
        """
        Initialize the model evaluator class

        Args:
            inferencer (type[InferencerBase]): The inferencer class to use for making predictions.
        Returns:
            None
        """
        super().__init__()

        self._inferencer = inferencer

    def _format_metrics(self, model: Any, metrics: dict[str, Any]) -> None:
        """
        Format the evaluation metrics into a nice table and print it to the console.

        Args:
            model (Any): The model being evaluated.
            metrics (dict[str, Any]): The evaluation metrics to format and print.
        Returns:
            None
        """

        cm = metrics["confusion_matrix"]
        tn, fp, fn, tp = cm.ravel()

        metrics_table = Table(
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            padding=(0, 2),
            show_edge=False,
        )
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Score", justify="right")
        for label, key in [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1", "f1"),
            ("ROC AUC", "roc_auc"),
        ]:
            metrics_table.add_row(label, f"{metrics[key]:.4f}")

        cm_table = Table(
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            padding=(0, 2),
            show_edge=False,
        )
        cm_table.add_column("", style="bold")
        cm_table.add_column("Pred 0", justify="right")
        cm_table.add_column("Pred 1", justify="right")
        cm_table.add_row("True 0", f"{tn}", f"{fp}")
        cm_table.add_row("True 1", f"{fn}", f"{tp}")

        layout = Table.grid(padding=(0, 6))
        layout.add_column(justify="right")
        layout.add_column(justify="left")
        layout.add_row(metrics_table, cm_table)

        title = Text(
            f"Evaluation — {model.__class__.__name__}",
            style="bold",
            justify="center",
        )
        console.print(Align.center(Group(title, Text(""), layout)))

    def run(
        self,
        model: Any,
        transformer: TransformBase,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Run the model evaluation step.

        Args:
            model (Any): The model to evaluate.
            transformer (ScalerBase): The scaler used to scale the test data.
            x_test (pd.DataFrame): The test features.
            y_test (pd.DataFrame): The test target.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        Returns:
            dict[str, Any]: The evaluation results, including the scaled test features,
                predictions, confusion matrix, accuracy, ROC AUC, precision, recall, and F1 score.
        """
        # Log the evaluation process and shapes of the inputs
        if verbose:
            console.section(title="Evaluating Model")
            logger.info(f"Evaluating model: {model.__class__.__name__}")
            logger.info(f"Evaluating x_test with shape: {x_test.shape}")
            logger.info(f"Evaluating y_test with shape: {y_test.shape}")

        # Scale the test data using the provided transformer
        x_test_scaled = transformer.transform(x_test)

        # Use the inferencer to make predictions and calculate probabilities (if supported)
        inferencer = self._inferencer()
        y_pred = inferencer.inference(model, x_test_scaled)

        try:
            y_pred_proba = inferencer.inference_proba(model, x_test_scaled)
            roc_auc = roc_auc_score(y_score=y_pred_proba, y_true=y_test)
        except NotImplementedError:
            logger.warning("Inference proba not implemented for this inferencer. ROC AUC will be set to 0.5.")
            roc_auc = None

        # Calculate the evaluation metrics
        results = {
            "x_test_scaled": x_test_scaled,
            "y_pred": y_pred,
            "confusion_matrix": confusion_matrix(y_pred=y_pred, y_true=y_test),
            "accuracy":  balanced_accuracy_score(y_pred=y_pred, y_true=y_test),
            "roc_auc":   roc_auc,
            "precision": precision_score(y_pred=y_pred, y_true=y_test),
            "recall":    recall_score(y_pred=y_pred, y_true=y_test),
            "f1":        f1_score(y_pred=y_pred, y_true=y_test),
        }

        # Format and print the evaluation metrics to the console
        if verbose:
            console.print()
            self._format_metrics(model, results)
            console.print()
            logger.info("Finished evaluating model.")

        return results