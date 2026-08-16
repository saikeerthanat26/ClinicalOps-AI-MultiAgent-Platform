from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "artifacts"
    / "readmission_xgboost.joblib"
)


class RiskService:
    def __init__(self) -> None:
        self._artifact = None
        self._explainer = None

    # --------------------------------------------------
    # Load saved model artifact
    # --------------------------------------------------

    def _load_artifact(
        self,
    ) -> dict[str, Any]:

        if self._artifact is not None:
            return self._artifact

        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                (
                    "Readmission model artifact was "
                    f"not found at {MODEL_FILE}. "
                    "Run: "
                    "python scripts/train_risk_model.py"
                )
            )

        self._artifact = joblib.load(
            MODEL_FILE
        )

        return self._artifact

    # --------------------------------------------------
    # Prepare features in training-column order
    # --------------------------------------------------

    def _prepare_dataframe(
        self,
        features: dict[str, Any],
    ) -> pd.DataFrame:

        artifact = self._load_artifact()

        feature_columns = artifact[
            "feature_columns"
        ]

        dataframe = pd.DataFrame(
            [
                {
                    column: features[column]
                    for column in feature_columns
                }
            ]
        )

        return dataframe

    # --------------------------------------------------
    # Lazily create SHAP explainer
    # --------------------------------------------------

    def _get_explainer(self):
        if self._explainer is not None:
            return self._explainer

        artifact = self._load_artifact()

        model = artifact["model"]

        self._explainer = shap.TreeExplainer(
            model
        )

        return self._explainer

    # --------------------------------------------------
    # Standard prediction
    # --------------------------------------------------

    def predict(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:

        artifact = self._load_artifact()

        model = artifact["model"]

        threshold = float(
            artifact[
                "decision_threshold"
            ]
        )

        dataframe = (
            self._prepare_dataframe(
                features
            )
        )

        probability = float(
            model.predict_proba(
                dataframe
            )[0][1]
        )

        predicted_readmission = (
            probability
            >= threshold
        )

        risk_level = self._risk_level(
            probability
        )

        return {
            "probability": round(
                probability,
                4,
            ),
            "probability_percent": round(
                probability * 100,
                2,
            ),
            "decision_threshold": round(
                threshold,
                4,
            ),
            "predicted_readmission": bool(
                predicted_readmission
            ),
            "risk_level": risk_level,
        }

    # --------------------------------------------------
    # Prediction + SHAP explanation
    # --------------------------------------------------

    def predict_with_explanation(
        self,
        features: dict[str, Any],
        top_n: int = 5,
    ) -> dict[str, Any]:

        prediction = self.predict(
            features
        )

        artifact = self._load_artifact()

        feature_columns = artifact[
            "feature_columns"
        ]

        dataframe = (
            self._prepare_dataframe(
                features
            )
        )

        explainer = (
            self._get_explainer()
        )

        shap_values = (
            explainer.shap_values(
                dataframe,
                check_additivity=True,
            )
        )

        values_array = np.asarray(
            shap_values
        )

        # XGBoost binary classifier normally
        # returns:
        #
        # (samples, features)
        #
        # This extra handling keeps the
        # service safer if an output dimension
        # appears in another model configuration.

        if values_array.ndim == 3:
            values_array = (
                values_array[
                    :,
                    :,
                    -1,
                ]
            )

        patient_values = (
            values_array[0]
        )

        factors = []

        for index, feature_name in enumerate(
            feature_columns
        ):

            shap_value = float(
                patient_values[index]
            )

            feature_value = float(
                dataframe.iloc[
                    0
                ][feature_name]
            )

            if shap_value > 0:
                impact = (
                    "increases_risk"
                )

            elif shap_value < 0:
                impact = (
                    "decreases_risk"
                )

            else:
                impact = "neutral"

            factors.append(
                {
                    "feature": (
                        feature_name
                    ),
                    "value": round(
                        feature_value,
                        4,
                    ),
                    "shap_value": round(
                        shap_value,
                        4,
                    ),
                    "impact": impact,
                    "_absolute_impact": abs(
                        shap_value
                    ),
                }
            )

        factors.sort(
            key=lambda factor: factor[
                "_absolute_impact"
            ],
            reverse=True,
        )

        top_factors = []

        for factor in factors[:top_n]:
            factor.pop(
                "_absolute_impact"
            )

            top_factors.append(
                factor
            )

        return {
            **prediction,
            "explanation_method": (
                "SHAP TreeExplainer"
            ),
            "shap_output_space": (
                "raw_margin_log_odds"
            ),
            "top_factors": (
                top_factors
            ),
        }

    # --------------------------------------------------
    # Human-friendly probability category
    # --------------------------------------------------

    @staticmethod
    def _risk_level(
        probability: float,
    ) -> str:

        if probability < 0.25:
            return "low"

        if probability < 0.50:
            return "moderate"

        if probability < 0.75:
            return "high"

        return "very_high"


risk_service = RiskService()