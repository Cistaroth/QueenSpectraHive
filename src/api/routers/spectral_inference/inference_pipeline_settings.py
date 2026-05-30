from pydantic import BaseModel

from modules.transformer.transformer_train import AudioTransformer

class InferencePipelineSettings(BaseModel):
    inference_pipeline: str
    inference_model: AudioTransformer = AudioTransformer()