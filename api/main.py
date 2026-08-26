import os
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.exceptions import (
    ChatbotUnavailableError,
    ConversationNotFoundError,
)
from api.v1.router import api_v1_router
from rag.retriever import get_retriever


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.retriever = get_retriever()
        app.state.retriever_error = None
    except Exception as exc:
        app.state.retriever = None
        app.state.retriever_error = str(exc)

    yield

    app.state.retriever = None


app = FastAPI(
    title=os.getenv(
        "API_TITLE",
        "Intelligent Knowledge Assistant API",
    ),
    version=os.getenv(
        "API_VERSION",
        "1.0.0",
    ),
    description=(
        "REST API for the Intelligent Knowledge "
        "Assistant chatbot."
    ),
    lifespan=lifespan,
)


cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "API_CORS_ORIGINS",
        "http://localhost:8501",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Request-ID",
    ],
)


@app.middleware("http")
async def request_id_middleware(
    request: Request,
    call_next,
):
    supplied_request_id = request.headers.get("X-Request-ID")

    try:
        request_id = str(
            UUID(supplied_request_id)
            if supplied_request_id
            else uuid4()
        )
    except ValueError:
        request_id = str(uuid4())

    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


def error_body(
    request: Request,
    code: str,
    message: str,
) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
        },
        "request_id": request.state.request_id,
    }


@app.exception_handler(ConversationNotFoundError)
async def conversation_not_found_handler(
    request: Request,
    exc: ConversationNotFoundError,
):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=error_body(
            request,
            code="CONVERSATION_NOT_FOUND",
            message=str(exc),
        ),
    )


@app.exception_handler(ChatbotUnavailableError)
async def chatbot_unavailable_handler(
    request: Request,
    exc: ChatbotUnavailableError,
):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=error_body(
            request,
            code="CHATBOT_UNAVAILABLE",
            message=(
                "The chatbot service is temporarily "
                "unavailable."
            ),
        ),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            **error_body(
                request,
                code="VALIDATION_ERROR",
                message="The request data is invalid.",
            ),
            "details": exc.errors(),
        },
    )


app.include_router(
    api_v1_router,
    prefix="/api/v1",
)
