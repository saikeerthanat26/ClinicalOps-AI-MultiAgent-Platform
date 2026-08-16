from pydantic import BaseModel, Field


class PatientQuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question about the selected synthetic patient",
    )


class PatientQuestionResponse(BaseModel):
    patient_id: str
    answer: str
    model: str
    grounded: bool