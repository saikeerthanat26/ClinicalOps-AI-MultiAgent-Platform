import json
from pathlib import Path
from typing import Any


FHIR_DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "fhir"
    / "clinical_bundle.json"
)


class FHIRService:
    def __init__(self) -> None:
        self.bundle = self._load_bundle()
        self.resources = [
            entry["resource"]
            for entry in self.bundle.get("entry", [])
            if "resource" in entry
        ]

    def _load_bundle(self) -> dict[str, Any]:
        with open(FHIR_DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    def get_patients(self) -> list[dict[str, Any]]:
        return [
            resource
            for resource in self.resources
            if resource.get("resourceType") == "Patient"
        ]

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        for patient in self.get_patients():
            if patient.get("id") == patient_id:
                return patient

        return None

    def get_patient_resources(
        self,
        patient_id: str,
        resource_type: str,
    ) -> list[dict[str, Any]]:
        patient_reference = f"Patient/{patient_id}"

        return [
            resource
            for resource in self.resources
            if resource.get("resourceType") == resource_type
            and resource.get("subject", {}).get("reference")
            == patient_reference
        ]

    def get_conditions(
        self,
        patient_id: str,
    ) -> list[dict[str, Any]]:
        return self.get_patient_resources(
            patient_id,
            "Condition",
        )

    def get_observations(
        self,
        patient_id: str,
    ) -> list[dict[str, Any]]:
        return self.get_patient_resources(
            patient_id,
            "Observation",
        )

    def get_medications(
        self,
        patient_id: str,
    ) -> list[dict[str, Any]]:
        return self.get_patient_resources(
            patient_id,
            "MedicationRequest",
        )

    def get_encounters(
        self,
        patient_id: str,
    ) -> list[dict[str, Any]]:
        return self.get_patient_resources(
            patient_id,
            "Encounter",
        )

    def get_patient_context(
        self,
        patient_id: str,
    ) -> dict[str, Any] | None:
        patient = self.get_patient(patient_id)

        if patient is None:
            return None

        return {
            "patient": patient,
            "conditions": self.get_conditions(patient_id),
            "observations": self.get_observations(patient_id),
            "medications": self.get_medications(patient_id),
            "encounters": self.get_encounters(patient_id),
        }



    def build_patient_context_text(
        self,
        patient_id: str,
    ) -> str | None:
        context = self.get_patient_context(patient_id)

        if context is None:
            return None

        patient = context["patient"]

        names = patient.get("name", [])
        official_name = "Unknown"

        if names:
            given = " ".join(names[0].get("given", []))
            family = names[0].get("family", "")
            official_name = f"{given} {family}".strip()

        lines = [
            "SYNTHETIC PATIENT RECORD",
            f"Patient ID: {patient.get('id', 'Unknown')}",
            f"Name: {official_name}",
            f"Gender: {patient.get('gender', 'Unknown')}",
            f"Birth Date: {patient.get('birthDate', 'Unknown')}",
            "",
            "ACTIVE CONDITIONS:",
        ]

        conditions = context["conditions"]

        if conditions:
            for condition in conditions:
                condition_name = (
                    condition.get("code", {})
                    .get("text", "Unknown condition")
                )
                lines.append(f"- {condition_name}")
        else:
            lines.append("- None documented")

        lines.append("")
        lines.append("CLINICAL OBSERVATIONS:")

        observations = context["observations"]

        if observations:
            for observation in observations:
                name = (
                    observation.get("code", {})
                    .get("text", "Unknown observation")
                )

                quantity = observation.get("valueQuantity", {})
                value = quantity.get("value", "Unknown")
                unit = quantity.get("unit", "")

                lines.append(
                    f"- {name}: {value} {unit}".strip()
                )
        else:
            lines.append("- None documented")

        lines.append("")
        lines.append("ACTIVE MEDICATIONS:")

        medications = context["medications"]

        if medications:
            for medication in medications:
                name = (
                    medication
                    .get("medicationCodeableConcept", {})
                    .get("text", "Unknown medication")
                )
                lines.append(f"- {name}")
        else:
            lines.append("- None documented")

        lines.append("")
        lines.append("ENCOUNTERS:")

        encounters = context["encounters"]

        if encounters:
            for encounter in encounters:
                encounter_type = (
                    encounter.get("class", {})
                    .get("display", "Unknown encounter")
                )

                reasons = encounter.get("reasonCode", [])

                reason = (
                    reasons[0].get("text", "No reason documented")
                    if reasons
                    else "No reason documented"
                )

                period = encounter.get("period", {})
                start = period.get("start", "Unknown")

                lines.append(
                    f"- {encounter_type}; "
                    f"reason: {reason}; "
                    f"start: {start}"
                )
        else:
            lines.append("- None documented")

        return "\n".join(lines)


fhir_service = FHIRService()