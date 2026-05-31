from pathlib import Path

import torch
import torch.nn as nn

from pipeline import ModelPipeline
from modules.inference.audio_saving import AudioSavingModule
from modules.lstm.lstm_inference import FusionLSTMInferenceModule
from modules.lstm.lstm_model import FusionLSTMModel

MODEL_PATH = Path(__file__).parents[4] / "trained_models" / "lstm" / "lstm_model.pth"

MODEL = FusionLSTMModel(
    lstm_layers=(40, 64, 2, 0.4),
    features_input_size=32,
    embeddings_model=(nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 32)),
    ff_hidden_size=64,
)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
MODEL.eval()

INFERENCE_PIPELINE = ModelPipeline(
    steps=[
        AudioSavingModule(),
        FusionLSTMInferenceModule(drop_column=["file name", "start_sec", "end_sec"]),
    ]
).add_context(
    step=1, context={"model": MODEL, "verbose": False}
).add_context(
    step=0, context={"verbose": False}
)