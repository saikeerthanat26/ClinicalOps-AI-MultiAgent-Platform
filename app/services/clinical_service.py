from app.services.fhir_service import fhir_service
from app.services.ollama_service import ollama_service


class ClinicalService:
    async def ask_patient(
        self,
        patient_id: str,
        question: str,
    ) -> str | None:
        context = fhir_service.build_patient_context_text(
            patient_id
        )

        if context is None:
            return None

        answer = await ollama_service.grounded_chat(
            message=question,
            context=context,
        )

        return answer


clinical_service = ClinicalService()