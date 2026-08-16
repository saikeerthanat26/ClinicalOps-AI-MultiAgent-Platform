import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.nlp.clinicalbert_service import (
    clinicalbert_service,
)


SYNTHETIC_CLINICAL_NOTE = """
Synthetic clinical note.

Patient with history of type 2 diabetes,
hypertension, and congestive heart failure.

Recent hospitalization was documented for
heart failure exacerbation.

Laboratory data includes HbA1c of 8.2 percent
and estimated glomerular filtration rate of 48.

Current documented medications include
metformin, lisinopril, and furosemide.

This note is fictional and intended only for
ClinicalOps AI software testing.
""".strip()


def main() -> None:

    print()
    print(
        "ClinicalOps ClinicalBERT Test"
    )

    print(
        "=" * 45
    )

    print()
    print(
        "Encoding synthetic clinical note..."
    )

    result = (
        clinicalbert_service.encode_note(
            SYNTHETIC_CLINICAL_NOTE
        )
    )

    print()
    print(
        f"Model               : "
        f"{result['model']}"
    )

    print(
        f"Token count         : "
        f"{result['token_count']}"
    )

    print(
        f"Embedding dimension : "
        f"{result['embedding_dimension']}"
    )

    print(
        f"Embedding norm      : "
        f"{result['embedding_norm']}"
    )

    print()

    print(
        "First 10 embedding values:"
    )

    print(
        result[
            "embedding_preview"
        ]
    )


if __name__ == "__main__":
    main()