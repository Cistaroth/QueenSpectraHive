from fastapi.responses import RedirectResponse
from fastapi.routing import APIRouter

router = APIRouter()

@router.get("/", description="Root endpoint that redirects to documentation.")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/interface/index.html")

