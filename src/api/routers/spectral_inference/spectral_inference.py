from typing import Optional

from fastapi import File, Form, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter

from api.routers.spectral_inference.inference_pipeline import INFERENCE_PIPELINE

router = APIRouter()

_WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}


@router.post(
    "/spectral-inference",
    description="Queen Bee Presence Classifier Endpoint for Inference on Spectral Data. Files must be in WAV format.",
    response_model=dict[str, bool | float],
    response_description=(
        "Returns a message indicating whether a queen bee was detected in the provided audio file. "
        "Also includes a boolean field for easier programmatic use."
    ),
)
async def spectral_inference(
    file: UploadFile = File(..., description="WAV audio recording of the hive"),
    hive_temp: Optional[float] = Form(None, description="Hive temperature (°C)"),
    hive_humidity: Optional[float] = Form(None, description="Hive relative humidity (%)"),
    hive_pressure: Optional[float] = Form(None, description="Hive atmospheric pressure (hPa)"),
    frames: Optional[int] = Form(None, description="Number of frames in the hive"),
    weather_temp: Optional[float] = Form(None, description="Outdoor temperature (°C)"),
    weather_humidity: Optional[float] = Form(None, description="Outdoor relative humidity (%)"),
    weather_pressure: Optional[float] = Form(None, description="Outdoor atmospheric pressure (hPa)"),
    wind_speed: Optional[float] = Form(None, description="Wind speed (m/s)"),
    cloud_coverage: Optional[float] = Form(None, description="Cloud coverage (%)"),
    date: Optional[str] = Form(None, description="Recording datetime in ISO 8601 format"),
    device: Optional[int] = Form(None, description="Device number (1 or 2)"),
    hive_number: Optional[int] = Form(None, description="Hive number (1-5)"),
) -> dict[str, str | bool | float]:
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

        print(pipeline_result)

    except Exception as e:
        print("Error during inference:", e)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during inference. Please try again later.",
        )

    return {
        "queen-detected": bool(result),
        "probability": probability,
    }
