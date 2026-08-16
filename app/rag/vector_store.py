import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


KNOWLEDGE_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "knowledge"
    / "knowledge_base.json"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self) -> None:
        self.documents = self._load_documents()

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.index = self._build_index()

    def _load_documents(self) -> list[dict[str, Any]]:
        with open(
            KNOWLEDGE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def _build_index(self) -> faiss.Index:
        texts = [
            f"{document['title']}\n{document['content']}"
            for document in self.documents
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings)

        return index

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        k = min(
            top_k,
            len(self.documents),
        )

        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            document = self.documents[
                int(index_position)
            ]

            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "source": document["source"],
                    "content": document["content"],
                    "score": round(
                        float(score),
                        4,
                    ),
                }
            )

        return results


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore()

    return _vector_store