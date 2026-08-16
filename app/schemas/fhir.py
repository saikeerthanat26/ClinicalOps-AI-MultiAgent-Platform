from typing import Any

from pydantic import BaseModel


class FHIRResourceListResponse(BaseModel):
    patient_id: str
    resource_type: str
    count: int
    resources: list[dict[str, Any]]


class PatientContextResponse(BaseModel):
    patient: dict[str, Any]
    conditions: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    medications: list[dict[str, Any]]
    encounters: list[dict[str, Any]]