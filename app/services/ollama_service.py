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


ollama_service = OllamaService()