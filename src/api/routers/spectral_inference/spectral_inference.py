from fastapi import File, Form, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel

from api.routers.spectral_inference.inference_pipeline import INFERENCE_PIPELINE

router = APIRouter()


class InferenceResponseModel(BaseModel):
    """
    Response model for the Queen Bee Presence Classifier Endpoint,
    containing the predicted presence of the queen bee and the associated probability.
    """

    queen_presence: bool
    probability: float


@router.post(
    "/spectral-inference",
    description="Queen Bee Presence Classifier Endpoint for Inference on Spectral Data. Files must be in WAV format.",
    response_model=InferenceResponseModel,
    response_description=(
        "Queen Bee Presence Classifier Response Model with Queen Presence (Boolean) and Queen Presence Probability (Float)"
    ),
)
async def spectral_inference(
    audio_file: UploadFile = File(..., description="WAV audio recording of the hive"),
    hive_temp: float = Form(..., description="Hive temperature (°C)"),
    hive_humidity: float = Form(..., description="Hive relative humidity (%)"),
    hive_pressure: float = Form(..., description="Hive atmospheric pressure (hPa)"),
    frames: int = Form(..., description="Number of frames in the hive"),
    weather_temp: float = Form(..., description="Outdoor temperature (°C)"),
    weather_humidity: float = Form(..., description="Outdoor relative humidity (%)"),
    weather_pressure: float = Form(
        ..., description="Outdoor atmospheric pressure (hPa)"
    ),
    wind_speed: float = Form(..., description="Wind speed (m/s)"),
    cloud_coverage: float = Form(..., description="Cloud coverage (%)"),
    date: str = Form(..., description="Recording datetime in ISO 8601 format"),
    device: int = Form(..., description="Device number (1 or 2)"),
    hive_number: int = Form(..., description="Hive number (1-5)"),
) -> InferenceResponseModel:

    content_type_check = audio_file.content_type in {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
    }

    filename_check = (audio_file.filename or "").lower().endswith(".wav")

    if not content_type_check and not filename_check:
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only WAV files are accepted."
        )

    tabular_data = {
        "hive_temp": hive_temp,
        "hive_humidity": hive_humidity,
        "hive_pressure": hive_pressure,
        "frames": frames,
        "weather_temp": weather_temp,
        "weather_humidity": weather_humidity,
        "weather_pressure": weather_pressure,
        "wind_speed": wind_speed,
        "cloud_coverage": cloud_coverage,
        "date": date,
        "device": device,
        "hive_number": hive_number,
    }

    try:
        INFERENCE_PIPELINE.add_context(
            step=0, context={"file": audio_file, "tabular_data": tabular_data}
        )
        pipeline_result = INFERENCE_PIPELINE.run()[-1].output

        result = pipeline_result["y_pred"][0]
        probability = float(pipeline_result["y_pred_proba"][0])

    except Exception as e:
        print(e)

        raise HTTPException(
            status_code=500,
            detail="An error occurred during inference. Please try again later.",
        )

    return InferenceResponseModel(queen_presence=result, probability=probability)
