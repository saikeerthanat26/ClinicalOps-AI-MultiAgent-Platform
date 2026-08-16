from typing import Any

import numpy as np
import torch

from transformers import (
    AutoModel,
    AutoTokenizer,
)


MODEL_NAME = (
    "emilyalsentzer/Bio_ClinicalBERT"
)

MAX_LENGTH = 512


class ClinicalBERTService:
    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None

    # --------------------------------------------------
    # Lazy model loading
    # --------------------------------------------------

    def _load_model(self) -> None:

        if (
            self._tokenizer is not None
            and self._model is not None
        ):
            return

        print(
            "Loading ClinicalBERT tokenizer..."
        )

        self._tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        print(
            "Loading ClinicalBERT model..."
        )

        self._model = (
            AutoModel.from_pretrained(
                MODEL_NAME
            )
        )

        self._model.eval()

        print(
            "ClinicalBERT loaded successfully."
        )

    # --------------------------------------------------
    # Mean pooling
    # --------------------------------------------------

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        expanded_mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(
                last_hidden_state.size()
            )
            .float()
        )

        summed_embeddings = torch.sum(
            last_hidden_state
            * expanded_mask,
            dim=1,
        )

        summed_mask = torch.clamp(
            expanded_mask.sum(
                dim=1
            ),
            min=1e-9,
        )

        return (
            summed_embeddings
            / summed_mask
        )

    # --------------------------------------------------
    # Internal embedding generation
    # --------------------------------------------------

    def _encode(
        self,
        note: str,
    ) -> tuple[np.ndarray, int]:

        self._load_model()

        tokenizer = self._tokenizer
        model = self._model

        encoded = tokenizer(
            note,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        token_count = int(
            encoded[
                "attention_mask"
            ].sum().item()
        )

        with torch.no_grad():

            output = model(
                **encoded
            )

            pooled_embedding = (
                self._mean_pool(
                    output.last_hidden_state,
                    encoded[
                        "attention_mask"
                    ],
                )
            )

        embedding = (
            pooled_embedding[
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        return (
            embedding,
            token_count,
        )

    # --------------------------------------------------
    # Public full embedding
    # --------------------------------------------------

    def get_embedding(
        self,
        note: str,
    ) -> np.ndarray:

        embedding, _ = (
            self._encode(
                note
            )
        )

        return embedding

    # --------------------------------------------------
    # Existing note encoding metadata
    # --------------------------------------------------

    def encode_note(
        self,
        note: str,
    ) -> dict[str, Any]:

        embedding, token_count = (
            self._encode(
                note
            )
        )

        embedding_norm = float(
            np.linalg.norm(
                embedding
            )
        )

        return {
            "model": MODEL_NAME,
            "token_count": (
                token_count
            ),
            "embedding_dimension": int(
                embedding.shape[0]
            ),
            "embedding_norm": round(
                embedding_norm,
                4,
            ),
            "embedding_preview": [
                round(
                    float(value),
                    4,
                )
                for value
                in embedding[:10]
            ],
        }


clinicalbert_service = (
    ClinicalBERTService()
)