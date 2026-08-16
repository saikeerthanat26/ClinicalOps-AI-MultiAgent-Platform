from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.ollama_service import ollama_service


router = APIRouter(
    prefix="/api/v1",
    tags=["Clinical AI"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = await ollama_service.chat(request.message)

        return ChatResponse(
            response=answer,
            model=settings.ollama_model,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local AI model unavailable: {str(exc)}",
        ) from exc