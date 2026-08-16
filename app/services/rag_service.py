from typing import Any

from app.rag.hybrid_retriever import (
    get_hybrid_retriever,
)
from app.services.ollama_service import (
    ollama_service,
)


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The retrieved knowledge base does not contain "
    "enough information to answer this question."
)


class RAGService:
    async def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict[str, Any]:

        retriever = get_hybrid_retriever()

        retrieval = retriever.search(
            query=question,
            top_k=top_k,
        )

        # ---------------------------------------
        # Relevance gate
        # ---------------------------------------

        if not retrieval["relevant"]:
            return {
                "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
                "retrieval_relevant": False,
                "generation_used": False,
                "max_dense_score": retrieval[
                    "max_dense_score"
                ],
                "sources": [],
            }

        retrieved_documents = retrieval[
            "results"
        ]

        # ---------------------------------------
        # Build grounded context
        # ---------------------------------------

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
                    "dense_score": document[
                        "dense_score"
                    ],
                    "bm25_score": document[
                        "bm25_score"
                    ],
                    "rrf_score": document[
                        "rrf_score"
                    ],
                    "reranker_score": document[
                        "reranker_score"
                    ],
                }
            )

        context = "\n\n---\n\n".join(
            context_sections
        )

        # ---------------------------------------
        # Generate only when evidence is relevant
        # ---------------------------------------

        answer = await ollama_service.rag_chat(
            question=question,
            context=context,
        )

        return {
            "answer": answer,
            "retrieval_relevant": True,
            "generation_used": True,
            "max_dense_score": retrieval[
                "max_dense_score"
            ],
            "sources": sources,
        }


rag_service = RAGService()