from typing import Any

from app.rag.vector_store import get_vector_store
from app.services.ollama_service import ollama_service


class RAGService:
    async def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict[str, Any]:

        vector_store = get_vector_store()

        retrieved_documents = vector_store.search(
            query=question,
            top_k=top_k,
        )

        context_sections = []

        sources = []

        for document in retrieved_documents:
            context_sections.append(
                (
                    f"[{document['id']}]\n"
                    f"Title: {document['title']}\n"
                    f"Source: {document['source']}\n"
                    f"Content: {document['content']}"
                )
            )

            sources.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "source": document["source"],
                    "score": document["score"],
                }
            )

        context = "\n\n---\n\n".join(
            context_sections
        )

        answer = await ollama_service.rag_chat(
            question=question,
            context=context,
        )

        return {
            "answer": answer,
            "sources": sources,
        }


rag_service = RAGService()