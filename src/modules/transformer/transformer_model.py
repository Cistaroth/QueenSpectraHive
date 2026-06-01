import torch
from torch import nn
from transformers import ASTForAudioClassification


class AudioTransformer(nn.Module):
    """
    This model loads a pretrained AST checkpoint from Hugging Face and adapts it
    to classify bee audio data.
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    ) -> None:
        """
        Initialize the model.

        Args:
            num_classes (int): Number of output units of the classification head.
                Defaults to 1.
            pretrained_model (str): Hugging Face model ID for AST feature extractor.
                Defaults to "MIT/ast-finetuned-audioset-10-10-0.4593".

        Returns:
            None
        """
        super().__init__()

        self.ast = ASTForAudioClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pretrained AST model.

        Args:
            x (torch.Tensor): Log-mel features of shape (B, time_frames, num_mel_bins).
        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        outputs = self.ast(x)
        return outputs.logits
