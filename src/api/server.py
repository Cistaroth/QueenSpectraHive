from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routers.root.root import router as root_router
from api.routers.spectral_inference.spectral_inference import (
    router as spectral_inference_router,
)
from config import config

app = FastAPI(
    title=config.NAME + " API Server",
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
)

app.mount(
    "/interface", StaticFiles(directory=Path(__file__).parent / "pages"), name="pages"
)
app.include_router(root_router)
app.include_router(spectral_inference_router)


def api_serve() -> None:
    """
    Main function to run the API server

    Args:
        None

    Returns:
        None
    """

    uvicorn.run(
        "api.server:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RUN_WITH_RELOAD,
    )


if __name__ == "__main__":
    api_serve()
