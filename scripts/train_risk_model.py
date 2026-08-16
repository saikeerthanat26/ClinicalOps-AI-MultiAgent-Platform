import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


# --------------------------------------------------
# Project root setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from app.ml.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


# --------------------------------------------------
# File locations
# --------------------------------------------------

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "synthetic_readmission_data.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
)

MODEL_FILE = (
    MODEL_DIR
    / "readmission_xgboost.joblib"
)

METRICS_FILE = (
    MODEL_DIR
    / "readmission_metrics.json"
)


# --------------------------------------------------
# Metrics helper
# --------------------------------------------------

def calculate_metrics(
    y_true,
    predictions,
    probabilities,
) -> dict:

    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "accuracy": round(
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "roc_auc": round(
            float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),
            4,
        ),
        "average_precision": round(
            float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),
            4,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def main() -> None:

    # --------------------------------------------------
    # Check dataset
    # --------------------------------------------------

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run this first:\n"
            "python scripts/generate_risk_data.py"
        )

    dataframe = pd.read_csv(
        DATA_FILE
    )

    print()
    print(
        f"Loaded dataset: "
        f"{DATA_FILE}"
    )

    print(
        f"Total records: "
        f"{len(dataframe)}"
    )

    # --------------------------------------------------
    # Features and target
    # --------------------------------------------------

    X = dataframe[
        FEATURE_COLUMNS
    ].copy()

    y = dataframe[
        TARGET_COLUMN
    ].copy()

    print(
        f"Overall positive rate: "
        f"{y.mean():.2%}"
    )

    # --------------------------------------------------
    # Train / validation / test split
    #
    # 70% training
    # 15% validation
    # 15% testing
    # --------------------------------------------------

    (
        X_train,
        X_remaining,
        y_train,
        y_remaining,
    ) = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    (
        X_validation,
        X_test,
        y_validation,
        y_test,
    ) = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.50,
        random_state=42,
        stratify=y_remaining,
    )

    # --------------------------------------------------
    # Handle class imbalance
    # --------------------------------------------------

    negative_count = int(
        (y_train == 0).sum()
    )

    positive_count = int(
        (y_train == 1).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "Training data contains no positive samples."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print()
    print(
        "Training class distribution"
    )

    print(
        "=" * 45
    )

    print(
        f"Negative samples: "
        f"{negative_count}"
    )

    print(
        f"Positive samples: "
        f"{positive_count}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------
    # Create XGBoost model
    # --------------------------------------------------

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    print()
    print(
        "Model training completed."
    )

    # --------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    # --------------------------------------------------
    # Find best decision threshold
    # using validation F1
    # --------------------------------------------------

    (
        precision_values,
        recall_values,
        thresholds,
    ) = precision_recall_curve(
        y_validation,
        validation_probabilities,
    )

    if len(thresholds) == 0:
        best_threshold = 0.50

    else:
        f1_values = (
            2
            * precision_values[:-1]
            * recall_values[:-1]
            / (
                precision_values[:-1]
                + recall_values[:-1]
                + 1e-10
            )
        )

        best_index = int(
            np.argmax(
                f1_values
            )
        )

        best_threshold = float(
            thresholds[
                best_index
            ]
        )

    # --------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    metrics = calculate_metrics(
        y_true=y_test,
        predictions=test_predictions,
        probabilities=test_probabilities,
    )

    # --------------------------------------------------
    # Additional metadata
    # --------------------------------------------------

    metrics.update(
        {
            "decision_threshold": round(
                float(
                    best_threshold
                ),
                4,
            ),
            "scale_pos_weight": round(
                float(
                    scale_pos_weight
                ),
                4,
            ),
            "training_samples": int(
                len(
                    X_train
                )
            ),
            "validation_samples": int(
                len(
                    X_validation
                )
            ),
            "test_samples": int(
                len(
                    X_test
                )
            ),
            "positive_rate_train": round(
                float(
                    y_train.mean()
                ),
                4,
            ),
            "positive_rate_validation": round(
                float(
                    y_validation.mean()
                ),
                4,
            ),
            "positive_rate_test": round(
                float(
                    y_test.mean()
                ),
                4,
            ),
        }
    )

    # --------------------------------------------------
    # Create model directory
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Save model artifact
    #
    # We save the model together with:
    # - threshold
    # - feature contract
    # - target name
    # --------------------------------------------------

    model_artifact = {
        "model": model,
        "decision_threshold": (
            best_threshold
        ),
        "feature_columns": (
            FEATURE_COLUMNS
        ),
        "target_column": (
            TARGET_COLUMN
        ),
    }

    joblib.dump(
        model_artifact,
        MODEL_FILE,
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------
    # Output report
    # --------------------------------------------------

    print()
    print(
        "ClinicalOps Synthetic "
        "Readmission Risk Model"
    )

    print(
        "=" * 45
    )

    print(
        f"Training samples   : "
        f"{metrics['training_samples']}"
    )

    print(
        f"Validation samples : "
        f"{metrics['validation_samples']}"
    )

    print(
        f"Test samples       : "
        f"{metrics['test_samples']}"
    )

    print()

    print(
        f"Train positive rate: "
        f"{metrics['positive_rate_train']:.2%}"
    )

    print(
        f"Val positive rate  : "
        f"{metrics['positive_rate_validation']:.2%}"
    )

    print(
        f"Test positive rate : "
        f"{metrics['positive_rate_test']:.2%}"
    )

    print()

    print(
        f"Decision threshold : "
        f"{metrics['decision_threshold']:.4f}"
    )

    print()

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision          : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall             : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC            : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Average Precision  : "
        f"{metrics['average_precision']:.4f}"
    )

    print()

    print(
        "Confusion Matrix"
    )

    print(
        "-" * 45
    )

    print(
        f"True Negatives  : "
        f"{metrics['true_negatives']}"
    )

    print(
        f"False Positives : "
        f"{metrics['false_positives']}"
    )

    print(
        f"False Negatives : "
        f"{metrics['false_negatives']}"
    )

    print(
        f"True Positives  : "
        f"{metrics['true_positives']}"
    )

    print()

    print(
        f"Model saved to:\n"
        f"{MODEL_FILE}"
    )

    print()

    print(
        f"Metrics saved to:\n"
        f"{METRICS_FILE}"
    )


if __name__ == "__main__":
    main()