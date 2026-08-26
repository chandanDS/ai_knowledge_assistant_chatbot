import os

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    ReadinessResponse,
)


router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API liveness",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="intelligent-knowledge-assistant",
        version=os.getenv(
            "API_VERSION",
            "1.0.0",
        ),
    )


@router.get(
    "/readiness",
    response_model=ReadinessResponse,
    summary="Check API readiness",
)
def readiness(
    request: Request,
) -> ReadinessResponse | JSONResponse:
    retriever_available = (
        getattr(
            request.app.state,
            "retriever",
            None,
        )
        is not None
    )

    response = ReadinessResponse(
        status=(
            "ready"
            if retriever_available
            else "not_ready"
        ),
        retriever_available=retriever_available,
    )

    if not retriever_available:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )

    return response
