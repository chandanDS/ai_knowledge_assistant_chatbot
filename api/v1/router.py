from fastapi import APIRouter

from api.v1.endpoints import (
    conversations,
    health,
)


api_v1_router = APIRouter()

api_v1_router.include_router(
    health.router,
    tags=["Health"],
)

api_v1_router.include_router(
    conversations.router,
    prefix="/conversations",
    tags=["Conversations"],
)
