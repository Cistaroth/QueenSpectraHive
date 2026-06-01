from typing import Any

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
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
        "x_test_scaled",
        "y_pred",
        "confusion_matrix",
        "accuracy",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "classification_report",
    }

    def __init__(
        self,
        inferencer: type[InferencerBase],
        class_names: list[str] | None = None,
    ) -> None:
        """
        Initialize the model evaluator class

        Args:
            inferencer (type[InferencerBase]): The inferencer class to use for making predictions.
            class_names (list[str] | None, optional): The class names to use in the classification report. Defaults to None.
        Returns:
            None
        """
        super().__init__()

        self._inferencer = inferencer
        self._class_names = class_names

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
        metrics_table.add_column("Metric", style="bold", justify="center")
        metrics_table.add_column("Score", justify="center")
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
        names = self._class_names or ["0", "1"]
        cm_table.add_column("", style="bold")
        cm_table.add_column(f"Pred {names[0]}", justify="right")
        cm_table.add_column(f"Pred {names[1]}", justify="right")
        cm_table.add_row(f"True {names[0]}", f"{tn}", f"{fp}")
        cm_table.add_row(f"True {names[1]}", f"{fn}", f"{tp}")

        report = metrics["classification_report"]

        report_table = Table(
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            padding=(0, 2),
            show_edge=False,
        )
        report_table.add_column("Class", style="bold")
        report_table.add_column("Precision", justify="right")
        report_table.add_column("Recall", justify="right")
        report_table.add_column("F1", justify="right")
        report_table.add_column("Support", justify="right")

        summary_keys = {"accuracy", "macro avg", "weighted avg"}
        class_keys = sorted(k for k in report if k not in summary_keys)

        for key in class_keys:
            row = report[key]
            report_table.add_row(
                str(key),
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1-score']:.4f}",
                str(int(row["support"])),
            )

        report_table.add_section()

        for label in ("macro avg", "weighted avg"):
            if label not in report:
                continue
            row = report[label]
            report_table.add_row(
                label,
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1-score']:.4f}",
                str(int(row["support"])),
            )

        table_width = 80
        metrics_table.min_width = table_width
        cm_table.min_width = table_width
        report_table.min_width = table_width

        title = Text(
            f"Evaluation — {model.__class__.__name__}",
            style="bold",
            justify="center",
        )
        console.print(
            Align.center(
                Group(
                    title,
                    Text(""),
                    metrics_table,
                    Text(""),
                    cm_table,
                    Text(""),
                    report_table,
                )
            )
        )

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

        # Some inferencers drop input rows
        kept_index = inferencer.last_kept_index
        if kept_index is not None:
            dropped = len(y_test) - len(kept_index)
            if dropped:
                logger.warning(
                    f"Inferencer dropped {dropped} row(s); evaluating on the "
                    f"remaining {len(kept_index)} of {len(y_test)} test samples."
                )

            y_test = y_test.iloc[kept_index]

        if len(y_pred) != len(y_test):
            raise ValueError(
                f"Prediction/label length mismatch after alignment: "
                f"{len(y_pred)} predictions vs {len(y_test)} labels."
            )

        try:
            y_pred_proba = inferencer.inference_proba(model, x_test_scaled)
            roc_auc = roc_auc_score(y_score=y_pred_proba, y_true=y_test)
        except NotImplementedError:
            logger.warning(
                "Inference proba not implemented for this inferencer. ROC AUC will be set to 0.5."
            )
            roc_auc = None

        # Calculate the evaluation metrics
        results = {
            "x_test_scaled": x_test_scaled,
            "y_pred": y_pred,
            "confusion_matrix": confusion_matrix(y_pred=y_pred, y_true=y_test),
            "accuracy": balanced_accuracy_score(y_pred=y_pred, y_true=y_test),
            "roc_auc": roc_auc,
            "precision": precision_score(y_pred=y_pred, y_true=y_test),
            "recall": recall_score(y_pred=y_pred, y_true=y_test),
            "f1": f1_score(y_pred=y_pred, y_true=y_test),
            "classification_report": classification_report(
                y_true=y_test,
                y_pred=y_pred,
                target_names=self._class_names,
                output_dict=True,
                zero_division=0,
            ),
        }

        # Format and print the evaluation metrics to the console
        if verbose:
            console.print()
            self._format_metrics(model, results)
            console.print()
            logger.info("Finished evaluating model.")

        return results
