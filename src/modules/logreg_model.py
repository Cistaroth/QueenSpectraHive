import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from modules.hyperparameter_tuning import TrainerBase, InferencerBase
from logger import console, logger

class LogRegTrainModule(TrainerBase):
    name="LogisticRegressionTraining"
    inputs = {"x_train_scaled", "y_train"}
    outputs = {"model"}

    def __init__(
        self, 
        **kwargs
    ) -> None:
        self._model_parameters = kwargs

        super().__init__()

    def run(
        self,
        x_train_scaled: pd.DataFrame,
        y_train: pd.DataFrame,
        verbose: bool = True
    ) -> None:
        if verbose:
            console.section(title="Training Logistic Regression Model")
            logger.info(f"Training on: {x_train_scaled.shape}")

        model = LogisticRegression(
            **self._model_parameters
        )
        
        model.fit(x_train_scaled, y_train)

        if verbose:
            logger.info(f"Finished training Logistic Regression Model. Model score: {model.score(x_train_scaled, y_train)}")
        
        return {
            "model": model
        }

    def train(
        self,
        x_train_scaled: pd.DataFrame,
        y_train: pd.DataFrame
    ) -> None:
        return self.run(x_train_scaled=x_train_scaled, y_train=y_train, verbose=False)["model"]

class LogRegInferenceModule(InferencerBase):
    name="LogisticRegressionEvaluation"
    inputs = {"model", "x_test_scaled", "y_test"}
    outputs = {"y_pred", "y_pred_proba"}

    def run(
        self,
        model: LogisticRegression,
        x_test_scaled: pd.DataFrame,
        verbose: bool = True
    ) -> None:
        if verbose:
            console.section(title="Evaluating Logistic Regression Model")
            logger.info(f"Evaluating on: {x_test_scaled.shape}")

        y_pred = model.predict(x_test_scaled)
        y_pred_proba = model.predict_proba(x_test_scaled)[:, 1]

        if verbose:
            logger.info(f"Finished evaluating Logistic Regression Model. Prediction shape: {y_pred.shape}")

        return {
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        }
        
    def inference(
        self,
        model: LogisticRegression,
        x_test_scaled: pd.DataFrame
    ) -> np.ndarray:
        return self.run(model=model, x_test_scaled=x_test_scaled, verbose=False)["y_pred"]
    
    def inference_proba(
        self,
        model: LogisticRegression,
        x_test_scaled: pd.DataFrame
    ) -> np.ndarray:
        return self.run(model=model, x_test_scaled=x_test_scaled, verbose=False)["y_pred_proba"]
        


