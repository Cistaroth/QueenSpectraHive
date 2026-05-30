from pathlib import Path

import torch
from pipeline import ModelPipeline
from modules.inference.audio_saving import AudioSavingModule
from modules.transformer.transformer_inference import TransformerInferenceModule
from modules.transformer.transformer_model import AudioTransformer

MODEL_PATH = Path(__file__).parents[4] / "trained_models" / "transformer" / "transformer_8.pth"
MODEL = AudioTransformer(
    num_classes = 1
)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

INFERENCE_PIPELINE = ModelPipeline(
    steps = [
        AudioSavingModule(),
        TransformerInferenceModule(),
    ]
).add_context(
    step = 1,
    context = {
        "model": MODEL
    }
)
