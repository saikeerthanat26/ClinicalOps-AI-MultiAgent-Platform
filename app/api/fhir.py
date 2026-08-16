from typing import Any

from fastapi import APIRouter, HTTPException

from app.schemas.fhir import (
    FHIRResourceListResponse,
    PatientContextResponse,
)
from app.services.fhir_service import fhir_service


router = APIRouter(
    prefix="/api/v1/fhir",
    tags=["FHIR"],
)


def ensure_patient_exists(patient_id: str) -> dict[str, Any]:
    patient = fhir_service.get_patient(patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found",
        )

    return patient


@router.get("/patients")
async def list_patients() -> list[dict[str, Any]]:
    return fhir_service.get_patients()


@router.get("/patients/{patient_id}")
async def get_patient(patient_id: str) -> dict[str, Any]:
    return ensure_patient_exists(patient_id)


@router.get(
    "/patients/{patient_id}/conditions",
    response_model=FHIRResourceListResponse,
)
async def get_conditions(
    patient_id: str,
) -> FHIRResourceListResponse:
    ensure_patient_exists(patient_id)

    resources = fhir_service.get_conditions(patient_id)

    return FHIRResourceListResponse(
        patient_id=patient_id,
        resource_type="Condition",
        count=len(resources),
        resources=resources,
    )


@router.get(
    "/patients/{patient_id}/observations",
    response_model=FHIRResourceListResponse,
)
async def get_observations(
    patient_id: str,
) -> FHIRResourceListResponse:
    ensure_patient_exists(patient_id)

    resources = fhir_service.get_observations(patient_id)

    return FHIRResourceListResponse(
        patient_id=patient_id,
        resource_type="Observation",
        count=len(resources),
        resources=resources,
    )


@router.get(
    "/patients/{patient_id}/medications",
    response_model=FHIRResourceListResponse,
)
async def get_medications(
    patient_id: str,
) -> FHIRResourceListResponse:
    ensure_patient_exists(patient_id)

    resources = fhir_service.get_medications(patient_id)

    return FHIRResourceListResponse(
        patient_id=patient_id,
        resource_type="MedicationRequest",
        count=len(resources),
        resources=resources,
    )


@router.get(
    "/patients/{patient_id}/encounters",
    response_model=FHIRResourceListResponse,
)
async def get_encounters(
    patient_id: str,
) -> FHIRResourceListResponse:
    ensure_patient_exists(patient_id)

    resources = fhir_service.get_encounters(patient_id)

    return FHIRResourceListResponse(
        patient_id=patient_id,
        resource_type="Encounter",
        count=len(resources),
        resources=resources,
    )


@router.get(
    "/patients/{patient_id}/context",
    response_model=PatientContextResponse,
)
async def get_patient_context(
    patient_id: str,
) -> PatientContextResponse:
    context = fhir_service.get_patient_context(patient_id)

    if context is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found",
        )

    return PatientContextResponse(**context)