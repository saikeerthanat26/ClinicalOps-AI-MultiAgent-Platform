import re

import httpx

from app.core.config import settings


SYSTEM_PROMPT = """
You are ClinicalOps AI, an educational healthcare AI assistant.

Your responsibilities:
- Explain healthcare and healthcare AI concepts clearly.
- Provide concise, professional, evidence-oriented responses.
- Do not claim to diagnose or treat patients.
- Clearly distinguish educational information from medical advice.
- Do not invent facts when information is uncertain.
- Give only the final answer to the user.

Do not expose internal reasoning or analysis.
""".strip()


def clean_model_response(content: str) -> str:
    """
    Remove reasoning traces that some local reasoning models may include
    before returning the final answer to the API consumer.
    """

    cleaned = content.strip()

    # Handle standard <think>...</think> blocks.
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    # Some model/runtime combinations return reasoning followed only
    # by a closing </think> tag.
    lower_cleaned = cleaned.lower()

    if "</think>" in lower_cleaned:
        position = lower_cleaned.rfind("</think>")
        cleaned = cleaned[position + len("</think>"):].strip()

    return cleaned


class OllamaService:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model


    

    async def chat(self, message: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

        raw_content = data["message"]["content"]

        return clean_model_response(raw_content)

    async def grounded_chat(
        self,
        message: str,
        context: str,
    ) -> str:
        grounded_system_prompt = """
You are ClinicalOps AI analyzing a SYNTHETIC healthcare record
for educational and software demonstration purposes.

STRICT RULES:

1. Use only information present in the provided patient context.
2. Do not invent diagnoses, medications, laboratory results,
   procedures, demographics, or clinical history.
3. If the requested information is not present, explicitly say:
   "The provided patient record does not contain that information."
4. Clearly distinguish documented facts from interpretation.
5. Do not provide treatment recommendations or claim to diagnose.
6. Keep the answer concise, professional, and evidence-oriented.
7. Never expose internal reasoning.
""".strip()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": grounded_system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "PATIENT CONTEXT:\n"
                        f"{context}\n\n"
                        "QUESTION:\n"
                        f"{message}"
                    ),
                },
            ],
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

        raw_content = data["message"]["content"]

        return clean_model_response(raw_content)


    async def rag_chat(
        self,
        question: str,
        context: str,
    ) -> str:
        system_prompt = """
You are ClinicalOps AI, an educational healthcare AI assistant.

You are answering questions using retrieved healthcare knowledge.

STRICT RULES:

1. Answer only from the retrieved evidence provided to you.
2. Do not invent healthcare facts that are not supported by the evidence.
3. If the evidence is insufficient, say:
   "The retrieved knowledge base does not contain enough information to answer this question."
4. Do not claim to diagnose or treat patients.
5. Do not provide personalized treatment recommendations.
6. Clearly distinguish documented evidence from general interpretation.
7. Keep responses concise and professional.
8. Do not expose internal reasoning.
""".strip()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        "RETRIEVED EVIDENCE:\n\n"
                        f"{context}\n\n"
                        "QUESTION:\n"
                        f"{question}"
                    ),
                },
            ],
            "think": False,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )

            response.raise_for_status()
            data = response.json()

        raw_content = data["message"]["content"]

        return clean_model_response(raw_content)


ollama_service = OllamaService()