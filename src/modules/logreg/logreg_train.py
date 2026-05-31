import pandas as pd
from sklearn.linear_model import LogisticRegression

from modules.model_bases import TrainerBase
from logger import console, logger

class LogRegTrainModule(TrainerBase):
    name="LogisticRegressionTraining"
    inputs = {"x_train", "y_train"}
    outputs = {"model"}

    def __init__(
        self, 
        **kwargs
    ) -> None:
        """
        Initialize the logistic regression trainer class

        Args:
            **kwargs: Keyword arguments to pass to the LogisticRegression model.
        Returns:
            None
        """
        super().__init__()

        self._model_parameters = kwargs

    def run(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        verbose: bool = True
    ) -> dict[str, LogisticRegression]:
        """
        Train a logistic regression model using the provided training data.

        Args:
            x_train (pd.DataFrame): The training features.
            y_train (pd.DataFrame): The training target variable.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        Returns:
            dict[str, LogisticRegression]: The trained logistic regression model.
        """

        # Log the training process and shapes of the inputs
        if verbose:
            console.section(title="Training Logistic Regression Model")
            logger.info(f"Training on: {x_train.shape}")

        # Initialize the logistic regression model with the provided parameters
        model = LogisticRegression(
            **self._model_parameters
        )

        # Fit the model to the training data
        model.fit(x_train, y_train)

        # Log the model score on the training data
        if verbose:
            logger.info(f"Finished training Logistic Regression Model. "
                        f"Model score: {model.score(x_train, y_train)}")
        
        return {
            "model": model
        }

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_val: pd.DataFrame,
        y_val: pd.DataFrame
    ) -> LogisticRegression:
        """
        Train a logistic regression model and return the trained model.

        Args:
            x_train (pd.DataFrame): The training features.
            y_train (pd.DataFrame): The training target variable.
        Returns:
            LogisticRegression: The trained logistic regression model.
        """
        return self.run(x_train=x_train, y_train=y_train, verbose=False)["model"]


        


