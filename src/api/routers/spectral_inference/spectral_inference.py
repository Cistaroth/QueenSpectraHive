from fastapi import Request, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.routing import APIRouter

from api.routers.spectral_inference.inference_pipeline import INFERENCE_PIPELINE

router = APIRouter()

@router.post(
    "/spectral-inference",
    description="Queen Bee Presence Classifier Endpoint for Inference on Spectral Data. Files must be in WAV format.",
    response_model=dict[str, str | bool | float],
    response_description="Returns a message indicating whether a queen bee"
        "was detected in the provided audio file. Also includes a boolean field for easier programmatic use.",
)
async def spectral_inference(request: Request) -> dict[str, str | bool | float]:
    form = await request.form()

    file = form.get("file")
    if not isinstance(file, UploadFile):
        raise HTTPException(status_code=400, detail="No audio file provided.")

    if file.content_type not in {"audio/wav", "audio/x-wav"} \
        or not (file.filename or "").lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only WAV files are accepted."
        )

    tabular_data = {k: str(v) for k, v in form.items() if k != "file"}

    try:
        INFERENCE_PIPELINE.add_context(step=0, context={"file": file, "tabular_data": tabular_data})
        pipeline_result = INFERENCE_PIPELINE.run()[-1].output

        print(pipeline_result)
        result = pipeline_result["y_pred"][0]
        probability = float(pipeline_result["y_pred_proba"][0])

    except Exception as e:
        print("Error during inference:", e)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during inference. Please try again later."
        )

    label = "Queen Bee Detected" if result else "No Queen Bee Detected"
    return {
        "message": label,
        "boolean": bool(result),
        "probability": probability,
    }
