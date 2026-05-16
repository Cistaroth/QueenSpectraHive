from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd
import numpy as np
from rich import box
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.align import Align
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from modules.bases import TrainerBase, InferencerBase, ScalerBase
from pipeline import ModelPipelineStep
from logger import console, logger
from config import config


@dataclass(frozen=True)
class FineTuningModelConfiguration:
    model_name: str
    model_train: type[TrainerBase]
    model_inference: type[InferencerBase]
    scaler: type[ScalerBase]
    hyperparameters: dict[str, list[Any]]

    metric: str
    folds: int = 5

    @property
    def hyperparameter_grid(self) -> list[dict[str, Any]]:
        keys = self.hyperparameters.keys()
        values = self.hyperparameters.values()

        return [dict(zip(keys, combo)) for combo in product(*values)]


class HyperparameterTuningStratifiedKFoldModule(ModelPipelineStep):
    name = "HyperparameterTuningStratifiedKFold"
    inputs = {"x_train", "y_train"}
    outputs = {"x_train_scaled", "scaler", "cv_results", "model"}

    def __init__(
        self,
        model_configuration: FineTuningModelConfiguration
    ) -> None:
        self._model_configuration = model_configuration

        self._metrics = {
            "accuracy":  accuracy_score,
            "f1":        f1_score,
            "precision": precision_score,
            "recall":    recall_score,
            "roc_auc":   roc_auc_score,
        }

        if self._model_configuration.metric not in self._metrics:
            supported_metrics = ", ".join(self._metrics.keys())
            raise ValueError(
                f"Metric {self._model_configuration.metric} not supported. "
                f"Supported metrics: {supported_metrics}"
            )

        super().__init__()

    def _format_parameters(self, parameters: dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in parameters.items())

    def _build_results_table(
        self,
        cv_results: list[dict[str, Any]],
        best_idx: int,
    ) -> Table:
        metric = self._model_configuration.metric

        table = Table(
            title=f"CV Results — {self._model_configuration.model_name}",
            box=box.SIMPLE_HEAVY,
            title_style="bold",
            header_style="bold cyan",
        )
        table.add_column("", justify="center", width=2)
        table.add_column("Configuration")
        table.add_column(f"Mean {metric}", justify="right")
        table.add_column("Std", justify="right")
        table.add_column(f"Per-fold {metric}", justify="right")

        for i, r in enumerate(cv_results):
            is_best = i == best_idx
            marker = "★" if is_best else ""
            folds_str = " ".join(f"{s:.3f}" for s in r["fold_scores"])
            style = "bold green" if is_best else None

            table.add_row(
                marker,
                self._format_parameters(r["model_parameter"]),
                f"{r['mean_score']:.4f}",
                f"±{r['std_score']:.4f}",
                folds_str,
                style=style,
            )

        return Align.center(table)

    def run(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        verbose: bool = True
    ) -> dict[str, Any]:
        cfg = self._model_configuration
        model_parameters = cfg.hyperparameter_grid
        metric_fn = self._metrics[cfg.metric]

        if verbose:
            console.section(title="Hyperparameter tuning with stratified k-fold")
            logger.info(f"Model:    {cfg.model_name}")
            logger.info(f"Configuration:  {len(model_parameters)}   |   Folds: {cfg.folds}   "
                        f"|   Metric: {cfg.metric}   |   Scaler: {cfg.scaler.__name__}")
            console.print()

        folds = StratifiedKFold(
            n_splits=cfg.folds,
            shuffle=True,
            random_state=config.SEED,
        )
        cv_results: list[dict[str, Any]] = []

        total_steps = len(model_parameters) * cfg.folds

        progress_columns = (
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeElapsedColumn(),
        )

        with Progress(*progress_columns, console=console, disable=not verbose, transient=True) as progress:
            task = progress.add_task("Tuning", total=total_steps)

            for idx, model_parameter in enumerate(model_parameters):
                progress.update(
                    task,
                    description=f"[{idx+1}/{len(model_parameters)}] {self._format_parameters(model_parameter)}",
                )

                fold_scores: list[float] = []
                for train_idx, val_idx in folds.split(x_train, y_train):
                    x_train_fold = x_train.iloc[train_idx]
                    x_val_fold = x_train.iloc[val_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    y_val_fold = y_train.iloc[val_idx]

                    scaler = cfg.scaler()
                    x_train_scaled, x_val_scaled = scaler.scale(x_train_fold, x_val_fold)

                    trainer = cfg.model_train(**model_parameter)
                    model = trainer.train(x_train_scaled, y_train_fold)

                    inferencer = cfg.model_inference()
                    y_pred = inferencer.inference(model, x_val_scaled)

                    score = metric_fn(y_val_fold, y_pred)
                    fold_scores.append(score)

                    progress.advance(task)

                cv_results.append({
                    "model_parameter": model_parameter,
                    "mean_score": float(np.mean(fold_scores)),
                    "std_score": float(np.std(fold_scores)),
                    "fold_scores": fold_scores,
                })

        best_idx = max(range(len(cv_results)), key=lambda i: cv_results[i]["mean_score"])
        best_result = cv_results[best_idx]
        best_model_parameter = best_result["model_parameter"]

        if verbose:
            console.print(self._build_results_table(cv_results, best_idx))
            console.print()
            logger.info("Starting refitting scaler and training final model on full training set.")

        final_scaler = cfg.scaler()
        x_train_scaled = final_scaler.fit_transform(x=x_train)

        final_trainer = cfg.model_train(**best_model_parameter)
        best_model = final_trainer.train(x_train_scaled, y_train)

        if verbose:
            logger.info("Finished hyperparameter tuning.")

        return {
            "x_train_scaled": x_train_scaled,
            "scaler": final_scaler,
            "model": best_model,
            "cv_results": cv_results,
        }