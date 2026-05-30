import torch
from torch import nn
from transformers import ASTForAudioClassification

from logger import logger

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
        unfreeze_last_n_layers: int = 0,
    ) -> None:
        """"
        Initialize the model.

        Args:
            num_classes (int): Number of classes. Defaults to 2.
            pretrained_model (str): Hugging Face model ID for AST feature extractor.
                Defaults to "MIT/ast-finetuned-audioset-10-10-0.4593".
            freeze_backbone (bool): When True, freeze the pretrained AST backbone and
                train only the classification head (plus any top blocks selected via
                ``unfreeze_last_n_layers``). This preserves the pretrained audio
                features and is far more sample-efficient on a small, imbalanced dataset,
                which prevents the model from collapsing to always predicting the
                majority class. Defaults to True.
            unfreeze_last_n_layers (int): When ``freeze_backbone`` is True, also unfreeze
                the top N transformer encoder blocks (plus the final encoder LayerNorm),
                so the highest-level features can adapt to bee audio. This is the middle
                ground between a (too-weak) linear probe on frozen AudioSet features and
                a (too-aggressive) full fine-tune that destabilises the backbone. These
                blocks should be trained at a small learning rate. 0 keeps the backbone
                fully frozen. Ignored when ``freeze_backbone`` is False. Defaults to 0.

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
        self._unfreeze_last_n = max(0, int(unfreeze_last_n_layers))
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

    def _encoder_blocks(self) -> nn.ModuleList | None:
        """
        Locate the AST transformer encoder blocks as an ``nn.ModuleList``, robust to
        naming differences across ``transformers`` versions.

        Newer versions expose them directly as ``audio_spectrogram_transformer.layers``;
        older versions nest them under ``audio_spectrogram_transformer.encoder.layer``.
        Operating on the module objects (rather than hard-coded parameter-name prefixes)
        means selective unfreezing keeps working if the library renames things.

        Returns:
            nn.ModuleList | None: The encoder blocks, or None if they cannot be found.
        """
        base = getattr(self.ast, "audio_spectrogram_transformer", None)
        if base is None:
            return None
        encoder = getattr(base, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            return encoder.layer
        if hasattr(base, "layers"):
            return base.layers
        return None

    def _final_layernorm(self) -> nn.Module | None:
        """
        Return the post-encoder LayerNorm that normalises the features feeding the head.

        Returns:
            nn.Module | None: The final encoder LayerNorm, or None if absent.
        """
        base = getattr(self.ast, "audio_spectrogram_transformer", None)
        return getattr(base, "layernorm", None) if base is not None else None

    def _apply_backbone_freeze(self) -> None:
        """
        Freeze the pretrained backbone, leaving the classification head trainable. When
        ``unfreeze_last_n_layers`` > 0, also unfreeze the top N encoder blocks (plus the
        final encoder LayerNorm) so the highest-level features can adapt to bee audio.

        Returns:
            None
        """
        # Freeze everything, then selectively re-enable gradients.
        for param in self.ast.parameters():
            param.requires_grad = False

        # The classification head always trains.
        for name, param in self.ast.named_parameters():
            if self._is_head(name):
                param.requires_grad = True

        if self._unfreeze_last_n <= 0:
            return

        blocks = self._encoder_blocks()
        if blocks is None or len(blocks) == 0:
            logger.warning(
                f"Could not locate AST encoder blocks; backbone stays fully frozen "
                f"despite unfreeze_last_n_layers={self._unfreeze_last_n}."
            )
            return

        n = min(self._unfreeze_last_n, len(blocks))
        for block in list(blocks)[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        # The post-encoder LayerNorm normalises the features the head consumes, so let
        # it adapt alongside the unfrozen blocks.
        final_ln = self._final_layernorm()
        if final_ln is not None:
            for param in final_ln.parameters():
                param.requires_grad = True

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