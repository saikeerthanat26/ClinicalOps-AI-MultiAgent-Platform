from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.clinical import (
    PatientQuestionRequest,
    PatientQuestionResponse,
)
from app.services.clinical_service import clinical_service


router = APIRouter(
    prefix="/api/v1/clinical",
    tags=["Grounded Clinical AI"],
)


@router.post(
    "/patients/{patient_id}/ask",
    response_model=PatientQuestionResponse,
)
async def ask_about_patient(
    patient_id: str,
    request: PatientQuestionRequest,
) -> PatientQuestionResponse:
    answer = await clinical_service.ask_patient(
        patient_id=patient_id,
        question=request.question,
    )

    if answer is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found",
        )

    return PatientQuestionResponse(
        patient_id=patient_id,
        answer=answer,
        model=settings.ollama_model,
        grounded=True,
    )