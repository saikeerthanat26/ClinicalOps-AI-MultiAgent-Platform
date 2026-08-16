import json
from pathlib import Path
from typing import Any

import numpy as np

from app.nlp.clinicalbert_service import (
    MODEL_NAME,
    clinicalbert_service,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

PATTERN_FILE = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
    / "clinical_patterns.json"
)

SEMANTIC_WEIGHT = 0.40
KEYWORD_WEIGHT = 0.60


class ClinicalPatternService:
    def __init__(self) -> None:
        self._patterns = None
        self._pattern_embeddings = None

    # --------------------------------------------------
    # Load reference patterns
    # --------------------------------------------------

    def _load_patterns(
        self,
    ) -> list[dict[str, Any]]:

        if self._patterns is not None:
            return self._patterns

        if not PATTERN_FILE.exists():
            raise FileNotFoundError(
                (
                    "Clinical pattern file not found: "
                    f"{PATTERN_FILE}"
                )
            )

        with open(
            PATTERN_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            self._patterns = json.load(
                file
            )

        return self._patterns

    # --------------------------------------------------
    # Normalize embedding
    # --------------------------------------------------

    @staticmethod
    def _normalize_embedding(
        embedding: np.ndarray,
    ) -> np.ndarray:

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            return embedding

        return embedding / norm

    # --------------------------------------------------
    # Clinical keyword matching
    # --------------------------------------------------

    @staticmethod
    def _keyword_match(
        note: str,
        keywords: list[str],
    ) -> tuple[float, list[str]]:

        normalized_note = (
            note.lower()
        )

        matched_keywords = []

        for keyword in keywords:

            if keyword.lower() in normalized_note:

                matched_keywords.append(
                    keyword
                )

        if not keywords:
            return (
                0.0,
                matched_keywords,
            )

        keyword_score = (
            len(matched_keywords)
            / len(keywords)
        )

        return (
            float(keyword_score),
            matched_keywords,
        )

    # --------------------------------------------------
    # Build reference ClinicalBERT embeddings
    # --------------------------------------------------

    def _build_pattern_embeddings(
        self,
    ) -> np.ndarray:

        if (
            self._pattern_embeddings
            is not None
        ):
            return (
                self._pattern_embeddings
            )

        patterns = (
            self._load_patterns()
        )

        embeddings = []

        print(
            "Building ClinicalBERT "
            "reference pattern embeddings..."
        )

        for pattern in patterns:

            embedding = (
                clinicalbert_service
                .get_embedding(
                    pattern["text"]
                )
            )

            embedding = (
                self._normalize_embedding(
                    embedding
                )
            )

            embeddings.append(
                embedding
            )

        self._pattern_embeddings = (
            np.vstack(
                embeddings
            )
        )

        print(
            "Clinical pattern embeddings "
            "cached successfully."
        )

        return (
            self._pattern_embeddings
        )

    # --------------------------------------------------
    # Min-max normalize semantic scores
    # --------------------------------------------------

    @staticmethod
    def _normalize_scores(
        scores: np.ndarray,
    ) -> np.ndarray:

        minimum = float(
            np.min(scores)
        )

        maximum = float(
            np.max(scores)
        )

        score_range = (
            maximum - minimum
        )

        if score_range < 1e-8:
            return np.ones_like(
                scores,
                dtype=np.float32,
            )

        normalized = (
            (scores - minimum)
            / score_range
        )

        return normalized.astype(
            np.float32
        )

    # --------------------------------------------------
    # Hybrid ClinicalBERT + concept matching
    # --------------------------------------------------

    def match_patterns(
        self,
        note: str,
        top_k: int = 3,
    ) -> dict[str, Any]:

        patterns = (
            self._load_patterns()
        )

        pattern_embeddings = (
            self._build_pattern_embeddings()
        )

        note_embedding = (
            clinicalbert_service
            .get_embedding(
                note
            )
        )

        note_embedding = (
            self._normalize_embedding(
                note_embedding
            )
        )

        # ----------------------------------------------
        # ClinicalBERT cosine similarity
        # ----------------------------------------------

        raw_semantic_scores = (
            pattern_embeddings
            @ note_embedding
        )

        normalized_semantic_scores = (
            self._normalize_scores(
                raw_semantic_scores
            )
        )

        results = []

        for index, pattern in enumerate(
            patterns
        ):

            (
                keyword_score,
                matched_keywords,
            ) = self._keyword_match(
                note=note,
                keywords=pattern.get(
                    "keywords",
                    [],
                ),
            )

            semantic_score = float(
                normalized_semantic_scores[
                    index
                ]
            )

            raw_cosine_similarity = float(
                raw_semantic_scores[
                    index
                ]
            )

            hybrid_score = (
                SEMANTIC_WEIGHT
                * semantic_score
                + KEYWORD_WEIGHT
                * keyword_score
            )

            results.append(
                {
                    "id": pattern[
                        "id"
                    ],
                    "label": pattern[
                        "label"
                    ],
                    "title": pattern[
                        "title"
                    ],
                    "clinicalbert_similarity": round(
                        raw_cosine_similarity,
                        4,
                    ),
                    "semantic_score": round(
                        semantic_score,
                        4,
                    ),
                    "keyword_score": round(
                        keyword_score,
                        4,
                    ),
                    "hybrid_score": round(
                        float(
                            hybrid_score
                        ),
                        4,
                    ),
                    "matched_keywords": (
                        matched_keywords
                    ),
                }
            )

        results.sort(
            key=lambda result: result[
                "hybrid_score"
            ],
            reverse=True,
        )

        k = min(
            top_k,
            len(results),
        )

        return {
            "model": MODEL_NAME,
            "semantic_weight": (
                SEMANTIC_WEIGHT
            ),
            "keyword_weight": (
                KEYWORD_WEIGHT
            ),
            "matches": results[:k],
        }


clinical_pattern_service = (
    ClinicalPatternService()
)