from fastapi import UploadFile
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter

from api.routers.spectral_inference.inference_pipeline import INFERENCE_PIPELINE
router = APIRouter()

@router.post(
    "/spectral-inference",
    description="Queen Bee Presence Classifier Endpoint for Inference on Spectral Data. Files must be in WAV format.",
    response_model=dict[str, str | bool],
    response_description="Returns a message indicating whether a queen bee" \
        "was detected in the provided audio file. Also includes a boolean field for easier programmatic use.",
)
async def spectral_inference(file: UploadFile) -> dict[str, str | bool]:
    # Validate correctness of file type
    if file.content_type not in {"audio/wav", "audio/x-wav"} \
        or not (file.filename or "").lower().endswith(".wav"):

        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only WAV files are accepted."
        )

    try:
        INFERENCE_PIPELINE.add_context(step=0, context={"file": file})
        result = INFERENCE_PIPELINE.run()[-1].output["result"]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred during inference. Please try again later."
        )

    result = "Queen Bee Detected" if result else "No Queen Bee Detected"
    return {
        "message": result,
        "boolean": result == "Queen Bee Detected"
    }