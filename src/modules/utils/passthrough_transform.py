from typing import Any
from modules.model_bases import TransformBase

class PassThroughTransformModule(TransformBase):
    """
    Transformer that simply passes through all data
    """
    def run(self, x: Any) -> Any:
        """
        Run the transformer

        Args:
            x (Any): The data to transform
        
        Returns:
            Any: The transformed data
        """
        return x
    
    def fit_transform(self, x: Any) -> Any:
        """
        Fit and transform the data

        Args:
            x (Any): The data to fit and transform
        
        Returns:
            Any: The transformed data
        """

        return x
    
    def transform(self, x: Any) -> Any:
        """
        Transform the data

        Args:
            x (Any): The data to transform
        
        Returns:
            Any: The transformed data
        """
        return x


