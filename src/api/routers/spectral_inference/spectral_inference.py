from pydantic import BaseModel
from fastapi import File, Form, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter

from api.routers.spectral_inference.inference_pipeline import INFERENCE_PIPELINE

router = APIRouter()

_WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}

class ResponseModel(BaseModel):
    queen_presence: bool
    probability: float


@router.post(
    "/spectral-inference",
    description="Queen Bee Presence Classifier Endpoint for Inference on Spectral Data. Files must be in WAV format.",
    response_model=ResponseModel,
    response_description=(
        "Queen Bee Presence Classifier Response Model with Queen Presence (Boolean) and Queen Presence Probability (Float)"
    ),
)
async def spectral_inference(
    file: UploadFile = File(..., description="WAV audio recording of the hive"),
    hive_temp: float = Form(None, description="Hive temperature (°C)"),
    hive_humidity: float = Form(None, description="Hive relative humidity (%)"),
    hive_pressure: float | None = Form(None, description="Hive atmospheric pressure (hPa)"),
    frames: int | None = Form(None, description="Number of frames in the hive"),
    weather_temp: float | None = Form(None, description="Outdoor temperature (°C)"),
    weather_humidity: float | None = Form(None, description="Outdoor relative humidity (%)"),
    weather_pressure: float | None = Form(None, description="Outdoor atmospheric pressure (hPa)"),
    wind_speed: float | None = Form(None, description="Wind speed (m/s)"),
    cloud_coverage: float | None = Form(None, description="Cloud coverage (%)"),
    date: str | None = Form(None, description="Recording datetime in ISO 8601 format"),
    device: int | None = Form(None, description="Device number (1 or 2)"),
    hive_number: int = Form(None, description="Hive number (1-5)"),
) -> ResponseModel:
    content_type_ok = file.content_type in _WAV_TYPES
    filename_ok = (file.filename or "").lower().endswith(".wav")
    if not content_type_ok and not filename_ok:
        raise HTTPException(status_code=400, detail="Invalid file type. Only WAV files are accepted.")

    tabular_data = {
        "hive_temp":        hive_temp,
        "hive_humidity":    hive_humidity,
        "hive_pressure":    hive_pressure,
        "frames":           frames,
        "weather_temp":     weather_temp,
        "weather_humidity": weather_humidity,
        "weather_pressure": weather_pressure,
        "wind_speed":       wind_speed,
        "cloud_coverage":   cloud_coverage,
        "date":             date,
        "device":           device,
        "hive_number":      hive_number,
    }

    try:
        INFERENCE_PIPELINE.add_context(step=0, context={"file": file, "tabular_data": tabular_data})
        pipeline_result = INFERENCE_PIPELINE.run()[-1].output

        result = pipeline_result["y_pred"][0]
        probability = float(pipeline_result["y_pred_proba"][0])

    except Exception as e:
        print("Error during inference:", e)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during inference. Please try again later.",
        )

    return ResponseModel(queen_presence=result, probability=probability)
