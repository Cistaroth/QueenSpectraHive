import inspect
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from rich import box
from rich.align import Align
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
)

from config import config
from logger import console, logger
from modules.hyperparameter_tuning.configuration import FineTuningConfiguration
from pipeline import ModelPipelineStep


class HyperparameterTuningStratifiedKFoldModule(ModelPipelineStep):
    name = "HyperparameterTuningStratifiedKFold"
    inputs = {"x_train", "y_train"}
    outputs = {"x_train_scaled", "transformer", "cv_results", "model"}

    def __init__(
        self,
        model_configuration: FineTuningConfiguration,
    ) -> None:
        """
        Initialize the hyperparameter tuning class

        Args:
            model_configuration (FineTuningModelConfiguration):
                The configuration for fine-tuning the model.
        Returns:
            None
        """
        super().__init__()

        self._model_configuration = model_configuration

        self._metrics = {
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
            "f1": f1_score,
            "f1_macro": partial(f1_score, average="macro"),
            "precision": precision_score,
            "recall": recall_score,
            "roc_auc": roc_auc_score,
        }

        if self._model_configuration.metric not in self._metrics:
            supported_metrics = ", ".join(self._metrics.keys())
            raise ValueError(
                f"Metric {self._model_configuration.metric} not supported. "
                f"Supported metrics: {supported_metrics}"
            )

    def _extract_groups(self, x_train: pd.DataFrame) -> np.ndarray | None:
        """
        Build a per-row group id from the configured group column.

        Args:
            x_train (pd.DataFrame): The training features.
        Returns:
            np.ndarray | None: Group label per row, or None if grouping is unavailable.
        """
        group_column = self._model_configuration.group_column
        if not group_column:
            return None
        cols = [
            c
            for c in x_train.columns
            if c == group_column or c.startswith(f"{group_column}_")
        ]
        if not cols:
            return None
        return x_train[cols].astype(str).agg("|".join, axis=1).to_numpy()

    def _train_param_names(self, trainer: Any) -> set[str]:
        """
        Return the set of parameter names accepted by a trainer's ``train`` method.

        Args:
            trainer (Any): A trainer base instance.
        Returns:
            set[str]: Parameter names of train base instance.
        """
        try:
            return set(inspect.signature(trainer.train).parameters)
        except (TypeError, ValueError):
            return set()

    def _get_varying_keys(self, configs: dict[str, Any]) -> set[str]:
        """
        Return only the keys whose values differ across at least two configs.

        Args:
            configs (list[dict[str, Any]]): A list of configuration dictionaries to analyze.
        Returns:
            set[str]: A set of keys that have different values across the provided configs.
        """
        if not configs:
            return set()
        return {key for key in configs.keys() if len(configs[key]) > 1}

    def _format_parameters(
        self,
        model_params: dict[str, Any],
        transformer_params: dict[str, Any],
        varying_model_keys: set[str],
        varying_transformer_keys: set[str],
    ) -> str:
        """
        Format only the non-constant parameters into a nice string for display in the results table.

        Args:
            model_params (dict[str, Any]): The model parameters to format.
            transformer_params (dict[str, Any]): The transformer parameters to format.
            varying_model_keys (set[str]): The model parameter keys that vary across configs.
            varying_transformer_keys (set[str]): The transformer parameter keys that vary across configs.
        Returns:
            str: A formatted string of the varying parameters and their values,
                or "(default)" if all parameters are constant.
        """
        parts = [
            f"{k}={v}" for k, v in model_params.items() if k in varying_model_keys
        ] + [
            f"{k}={v}"
            for k, v in transformer_params.items()
            if k in varying_transformer_keys
        ]
        return ", ".join(parts) if parts else "(default)"

    def _build_results_table(
        self,
        cv_results: list[dict[str, Any]],
        best_idx: int,
        varying_model_keys: set[str],
        varying_transformer_keys: set[str],
    ) -> Align:
        """
        Build a rich table to display the cross-validation results in the console.

        Args:
            cv_results (list[dict[str, Any]]): The cross-validation results to display.
            best_idx (int): The index of the best result in the cv_results list.
            varying_model_keys (set[str]): Model parameter keys that vary across configs.
            varying_transformer_keys (set[str]): Transformer parameter keys that vary across configs.
        Returns:
            Align: A rich Align object containing the results table, centered in the console.
        """
        metric = self._model_configuration.metric

        # Build table headers
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

        # Build rows for each result
        for i, r in enumerate(cv_results):
            is_best = i == best_idx
            marker = "★" if is_best else ""
            folds_str = " ".join(f"{s:.3f}" for s in r["fold_scores"])
            style = "bold green" if is_best else None

            table.add_row(
                marker,
                self._format_parameters(
                    r["model_parameter"],
                    r["transformer_parameter"],
                    varying_model_keys,
                    varying_transformer_keys,
                ),
                f"{r['mean_score']:.4f}",
                f"±{r['std_score']:.4f}",
                folds_str,
                style=style,
            )

        return Align.center(table)

    def run(
        self, x_train: pd.DataFrame, y_train: pd.DataFrame, verbose: bool = True
    ) -> dict[str, Any]:
        """
        Run the hyperparameter tuning process with stratified k-fold cross-validation.
        Args:
            x_train (pd.DataFrame): The training features.
            y_train (pd.DataFrame): The training target.
            verbose (bool, optional): Whether to print progress and
                results to the console. Defaults to True.
        Returns:
            dict[str, Any]: A dictionary containing the scaled training features,
                the fitted transformer, the cross-validation results,
                and the best model trained on the full training set.
        """
        # Get the model configuration and metric function
        cfg = self._model_configuration
        model_grid = cfg.model_hyperparameter_grid
        transformer_grid = cfg.transformer_hyperparameter_grid
        metric_fn = self._metrics[cfg.metric]

        # Get keys that will be displayed
        varying_model_keys = self._get_varying_keys(cfg.model_hyperparameters)
        varying_transformer_keys = self._get_varying_keys(
            cfg.transformer_hyperparameters
        )

        # Print the tuning configuration to the console
        if verbose:
            console.section(title="Hyperparameter tuning with stratified k-fold")
            logger.info(f"Model:    {cfg.model_name}")
            logger.info(
                f"Configuration:  {len(model_grid) * len(transformer_grid)}   |   Folds: {cfg.folds}   "
                f"|   Metric: {cfg.metric}   |   Transformer: {cfg.transformer.__name__}"
            )
            console.print()

        groups = self._extract_groups(x_train)
        n_groups = len(np.unique(groups)) if groups is not None else 0
        use_grouped_cv = groups is not None and n_groups >= cfg.folds

        folds = StratifiedKFold(
            n_splits=cfg.folds,
            shuffle=True,
            random_state=config.SEED,
        )

        cv_results: list[dict[str, Any]] = []

        # Build progress bar for tuning process
        total_steps = len(model_grid) * len(transformer_grid) * cfg.folds
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

        # Run the tuning process with a progress bar
        combo_idx = 0
        total_combos = len(model_grid) * len(transformer_grid)

        with Progress(
            *progress_columns, console=console, disable=not verbose, transient=True
        ) as progress:
            task = progress.add_task("Tuning", total=total_steps)

            for model_parameter in model_grid:
                for transformer_parameter in transformer_grid:
                    combo_idx += 1
                    progress.update(
                        task,
                        description=(
                            f"[{combo_idx}/{total_combos}] "
                            + self._format_parameters(
                                model_parameter,
                                transformer_parameter,
                                varying_model_keys,
                                varying_transformer_keys,
                            )
                        ),
                    )

                    fold_scores = []
                    for train_idx, val_idx in folds.split(
                        x_train, y_train, groups=groups if use_grouped_cv else None
                    ):
                        # Get the training and validation folds
                        x_train_fold = x_train.iloc[train_idx]
                        x_val_fold = x_train.iloc[val_idx]
                        y_train_fold = y_train.iloc[train_idx]
                        y_val_fold = y_train.iloc[val_idx]

                        # Fit the transformer on the training fold and
                        # transform both the training and validation folds
                        transformer = cfg.transformer(**transformer_parameter)
                        x_train_transformed = transformer.fit_transform(x_train_fold)
                        x_val_transformed = transformer.transform(x_val_fold)

                        # Apply resampler
                        if cfg.resampler is not None:
                            resampler = cfg.resampler(**cfg.resampler_parameters)
                            resampled = resampler.run(
                                x_train_transformed, y_train_fold, verbose=False
                            )
                            x_train_transformed = resampled["x_train"]
                            y_train_fold = resampled["y_train"]

                        # Train the model on the training fold and evaluate it on the validation fold
                        trainer = cfg.model_train(**model_parameter)
                        fold_kwargs = (
                            {"save_model": False}
                            if "save_model" in self._train_param_names(trainer)
                            else {}
                        )
                        model = trainer.train(
                            x_train_transformed,
                            y_train_fold,
                            x_val_transformed,
                            y_val_fold,
                            **fold_kwargs,
                        )

                        inferencer = cfg.model_inference()
                        y_pred = inferencer.inference(model, x_val_transformed)

                        # Calculate the metric score for the current fold and store it
                        score = metric_fn(y_val_fold, y_pred)
                        fold_scores.append(score)

                        progress.advance(task)

                    cv_results.append(
                        {
                            "model_parameter": model_parameter,
                            "transformer_parameter": transformer_parameter,
                            "mean_score": float(np.mean(fold_scores)),
                            "std_score": float(np.std(fold_scores)),
                            "fold_scores": fold_scores,
                        }
                    )

        # Find the best model parameters based on the mean score across folds
        best_idx = max(
            range(len(cv_results)), key=lambda i: cv_results[i]["mean_score"]
        )
        best_result = cv_results[best_idx]

        # Print the cross-validation results and the best model parameters to the console
        if verbose:
            console.print(
                self._build_results_table(
                    cv_results, best_idx, varying_model_keys, varying_transformer_keys
                )
            )
            console.print()
            logger.info("Starting refitting transformer and training the final model.")

        final_transformer = cfg.transformer(**best_result["transformer_parameter"])
        final_trainer = cfg.model_train(**best_result["model_parameter"])
        train_params = self._train_param_names(final_trainer)
        save_kwargs = {"save_model": True} if "save_model" in train_params else {}

        x_train_transformed = final_transformer.fit_transform(x=x_train)
        y_refit = y_train

        if cfg.resampler is not None:
            resampler = cfg.resampler(**cfg.resampler_parameters)
            resampled = resampler.run(x_train_transformed, y_train, verbose=False)
            x_train_transformed = resampled["x_train"]
            y_refit = resampled["y_train"]

        best_model = final_trainer.train(
            x_train_transformed, y_refit, None, None, **save_kwargs
        )

        if verbose:
            logger.info("Finished hyperparameter tuning.")

        return {
            "x_train_transformed": x_train_transformed,
            "transformer": final_transformer,
            "model": best_model,
            "cv_results": cv_results,
        }
