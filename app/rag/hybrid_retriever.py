import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.rag.vector_store import get_vector_store


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K = 60

MIN_DENSE_SCORE = 0.25


class HybridRetriever:
    def __init__(self) -> None:
        self.vector_store = get_vector_store()

        self.documents = self.vector_store.documents

        self.tokenized_documents = [
            self._tokenize(
                f"{document['title']} {document['content']}"
            )
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

        self.reranker = CrossEncoder(
            RERANKER_MODEL
        )

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(query)

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in ranked_indices:
            document = self.documents[
                int(index)
            ]

            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "source": document["source"],
                    "content": document["content"],
                    "bm25_score": round(
                        float(scores[index]),
                        4,
                    ),
                }
            )

        return results

    def _dense_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
        )

        return [
            {
                **result,
                "dense_score": result["score"],
            }
            for result in results
        ]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[dict[str, Any]],
        dense_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(
            bm25_results,
            start=1,
        ):
            document_id = result["id"]

            if document_id not in fused:
                fused[document_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "dense_score": 0.0,
                }

            fused[document_id]["bm25_score"] = (
                result["bm25_score"]
            )

            fused[document_id]["rrf_score"] += (
                1 / (RRF_K + rank)
            )

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            document_id = result["id"]

            if document_id not in fused:
                fused[document_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "bm25_score": 0.0,
                }

            fused[document_id]["dense_score"] = (
                result["dense_score"]
            )

            fused[document_id]["rrf_score"] += (
                1 / (RRF_K + rank)
            )

        return sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> dict[str, Any]:

        candidate_k = min(
            max(top_k * 2, 5),
            len(self.documents),
        )

        bm25_results = self._bm25_search(
            query=query,
            top_k=candidate_k,
        )

        dense_results = self._dense_search(
            query=query,
            top_k=candidate_k,
        )

        max_dense_score = max(
            (
                result["dense_score"]
                for result in dense_results
            ),
            default=0.0,
        )

        fused_results = (
            self._reciprocal_rank_fusion(
                bm25_results=bm25_results,
                dense_results=dense_results,
            )
        )

        candidate_results = fused_results[
            :candidate_k
        ]

        query_document_pairs = [
            (
                query,
                (
                    f"{result['title']}. "
                    f"{result['content']}"
                ),
            )
            for result in candidate_results
        ]

        reranker_scores = self.reranker.predict(
            query_document_pairs
        )

        for result, score in zip(
            candidate_results,
            reranker_scores,
        ):
            result["reranker_score"] = round(
                float(score),
                4,
            )

            result["rrf_score"] = round(
                float(result["rrf_score"]),
                6,
            )

        reranked_results = sorted(
            candidate_results,
            key=lambda item: item[
                "reranker_score"
            ],
            reverse=True,
        )

        relevant = (
            max_dense_score
            >= MIN_DENSE_SCORE
        )

        return {
            "relevant": relevant,
            "max_dense_score": round(
                float(max_dense_score),
                4,
            ),
            "results": reranked_results[
                :top_k
            ],
        }


_hybrid_retriever: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever

    if _hybrid_retriever is None:
        _hybrid_retriever = (
            HybridRetriever()
        )

    return _hybrid_retriever