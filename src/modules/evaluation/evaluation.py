from typing import Any

import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
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

from modules.model_bases import InferencerBase, ScalerBase
from pipeline import ModelPipelineStep
from logger import console, logger


class ModelEvaluatorModule(ModelPipelineStep):
    name = "ModelEvaluator"
    inputs = {"model", "scaler", "x_test", "y_test"}
    outputs = {
        "x_test_scaled", "y_pred", "confusion_matrix",
        "accuracy", "roc_auc", "precision", "recall", "f1",
    }

    def __init__(self, inferencer: type[InferencerBase]) -> None:
        self._inferencer = inferencer
        super().__init__()

    def _format_metrics(self, model: Any, metrics: dict[str, Any]) -> None:
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
        scaler: ScalerBase,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, Any]:
        if verbose:
            console.section(title="Evaluating Model")
            logger.info(f"Evaluating model: {model.__class__.__name__}")
            logger.info(f"Evaluating x_test with shape: {x_test.shape}")
            logger.info(f"Evaluating y_test with shape: {y_test.shape}")

        x_test_scaled = scaler.transform(x_test)

        inferencer = self._inferencer()
        y_pred = inferencer.inference(model, x_test_scaled)
        y_pred_proba = inferencer.inference_proba(model, x_test_scaled)

        results = {
            "x_test_scaled": x_test_scaled,
            "y_pred": y_pred,
            "confusion_matrix": confusion_matrix(y_pred=y_pred, y_true=y_test),
            "accuracy":  accuracy_score(y_pred=y_pred, y_true=y_test),
            "roc_auc":   roc_auc_score(y_score=y_pred_proba, y_true=y_test),
            "precision": precision_score(y_pred=y_pred, y_true=y_test),
            "recall":    recall_score(y_pred=y_pred, y_true=y_test),
            "f1":        f1_score(y_pred=y_pred, y_true=y_test),
        }

        if verbose:
            console.print()
            self._format_metrics(model, results)
            console.print()
            logger.info("Finished evaluating model.")

        return results