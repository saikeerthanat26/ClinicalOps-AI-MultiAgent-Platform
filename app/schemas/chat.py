from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Message sent to ClinicalOps AI",
    )


class ChatResponse(BaseModel):
    response: str
    model: str