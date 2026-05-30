import torch
from torch import nn
from transformers import ASTForAudioClassification

class AudioTransformer(nn.Module):
    """
    This model loads a pretrained AST checkpoint from Hugging Face and adapts it
    for two classes for on bee audio data.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        freeze_backbone: bool = True,
    ) -> None:
        """"
        Initialize the model.

        Args:
            num_classes (int): Number of classes. Defaults to 2.
            pretrained_model (str): Hugging Face model ID for AST feature extractor.
                Defaults to "MIT/ast-finetuned-audioset-10-10-0.4593".
            freeze_backbone (bool): When True, freeze the pretrained AST backbone and
                train only the classification head. This preserves the pretrained audio
                features and is far more sample-efficient on a small, imbalanced dataset,
                which prevents the model from collapsing to always predicting the
                majority class. Defaults to True.

        Returns:
            None
        """
        super().__init__()

        self.ast = ASTForAudioClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        self._freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._apply_backbone_freeze()

    @staticmethod
    def _is_head(param_name: str) -> bool:
        """
        Return whether a parameter (named relative to ``self.ast``) belongs to the
        classification head rather than the pretrained backbone.

        The AST head is ``classifier`` (a LayerNorm + Linear); everything else lives
        under ``audio_spectrogram_transformer`` and is considered backbone.

        Args:
            param_name (str): Parameter name from ``self.ast.named_parameters()``.
        Returns:
            bool: True if the parameter is part of the classification head.
        """
        return param_name.startswith("classifier")

    def _apply_backbone_freeze(self) -> None:
        """
        Freeze every backbone parameter, leaving only the classification head trainable.

        Returns:
            None
        """
        for name, param in self.ast.named_parameters():
            param.requires_grad = self._is_head(name)

    def parameter_groups(
        self,
        head_lr: float,
        backbone_lr: float,
    ) -> list[dict]:
        """
        Build optimizer parameter groups so the head and the backbone can use
        different learning rates.

        Only trainable parameters are included, so when the backbone is frozen this
        returns a single head group. When the backbone is unfrozen, it returns two
        groups: a head group at ``head_lr`` and a (typically much smaller) backbone
        group at ``backbone_lr``.

        Args:
            head_lr (float): Learning rate for the classification head.
            backbone_lr (float): Learning rate for the pretrained backbone (ignored
                when the backbone is frozen).
        Returns:
            list[dict]: Parameter groups suitable for a torch optimizer.
        """
        head_params: list = []
        backbone_params: list = []
        for name, param in self.ast.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_head(name):
                head_params.append(param)
            else:
                backbone_params.append(param)

        groups: list[dict] = []
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        return groups

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