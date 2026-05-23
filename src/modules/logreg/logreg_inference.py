import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from modules.model_bases import InferencerBase
from logger import console, logger

class LogRegInferenceModule(InferencerBase):
    name = "LogisticRegressionEvaluation"
    inputs = {"model", "x_test", "y_test"}
    outputs = {"y_pred", "y_pred_proba"}

    def run(
        self,
        model: LogisticRegression,
        x_test: pd.DataFrame,
        verbose: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Run inference using the provided model and test data.

        Args:
            model (LogisticRegression): The trained logistic regression model.
            x_test_scaled (pd.DataFrame): The scaled test features.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        Returns:
            dict[str, np.ndarray]: The predictions and predicted probabilities.
        """
        # Log the inference process and shapes of the inputs
        if verbose:
            console.section(title="Evaluating Logistic Regression Model")
            logger.info(f"Evaluating on: {x_test.shape}")

        # Make predictions and calculate predicted probabilities
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]

        # Log the shapes of the outputs
        if verbose:
            logger.info(
                f"Finished evaluating Logistic Regression Model. Prediction shape: {y_pred.shape}"
            )

        # Return the predictions and predicted probabilities
        return {
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        }

    def inference(
        self,
        model: LogisticRegression,
        x_test: pd.DataFrame
    ) -> np.ndarray:
        """"
        Run inference to get predictions using the provided model and test data.

        Args:
            model (LogisticRegression): The trained logistic regression model.
            x_test_scaled (pd.DataFrame): The scaled test features.
        Returns:
            np.ndarray: The predicted class labels.
        """
        return self.run(model=model, x_test=x_test, verbose=False)[
            "y_pred"
        ]

    def inference_proba(
        self,
        model: LogisticRegression,
        x_test: pd.DataFrame
    ) -> np.ndarray:
        """
        Run inference to get predicted probabilities using the provided model and test data.

        Args:
            model (LogisticRegression): The trained logistic regression model.
            x_test_scaled (pd.DataFrame): The scaled test features.
        Returns:
            np.ndarray: The predicted probabilities for the positive class.
        """
        return self.run(model=model, x_test=x_test, verbose=False)[
            "y_pred_proba"
        ]
