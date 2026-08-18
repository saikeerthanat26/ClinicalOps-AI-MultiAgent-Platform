import asyncio
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


from mcp import Client

from app.mcp_server import (
    mcp,
)


# ---------------------------------------------------------
# Pretty printing helper
# ---------------------------------------------------------

def print_result(
    title: str,
    result,
) -> None:

    print()
    print(
        "=" * 60
    )

    print(title)

    print(
        "=" * 60
    )

    print(
        f"is_error: "
        f"{result.is_error}"
    )

    print()

    print(
        "structured_content:"
    )

    print(
        result.structured_content
    )


# ---------------------------------------------------------
# MCP test client
# ---------------------------------------------------------

async def main() -> None:

    async with Client(
        mcp
    ) as client:

        # ------------------------------------------------
        # Discover tools
        # ------------------------------------------------

        tool_list = (
            await client.list_tools()
        )

        tool_names = [
            tool.name
            for tool
            in tool_list.tools
        ]

        print()
        print(
            "ClinicalOps MCP Server"
        )

        print(
            "=" * 60
        )

        print(
            "Discovered MCP tools:"
        )

        for tool_name in tool_names:

            print(
                f"- {tool_name}"
            )

        # ------------------------------------------------
        # Tool 1:
        # FHIR
        # ------------------------------------------------

        patient_result = (
            await client.call_tool(
                "get_patient_context",
                {
                    "patient_id": "P001",
                },
            )
        )

        print_result(
            "FHIR MCP TOOL RESULT",
            patient_result,
        )

        # ------------------------------------------------
        # Tool 2:
        # Hybrid RAG retrieval
        # ------------------------------------------------

        search_result = (
            await client.call_tool(
                "search_healthcare_knowledge",
                {
                    "query": (
                        "How does HCC risk "
                        "adjustment affect "
                        "Medicare Advantage "
                        "payments?"
                    ),
                    "top_k": 3,
                },
            )
        )

        print_result(
            "RAG MCP TOOL RESULT",
            search_result,
        )

        # ------------------------------------------------
        # Tool 3:
        # Predictive risk model
        # ------------------------------------------------

        risk_result = (
            await client.call_tool(
                "predict_readmission_risk",
                {
                    "features": {
                        "age": 82,
                        "prior_admissions_12m": 4,
                        "length_of_stay_days": 10,
                        "chronic_condition_count": 6,
                        "medication_count": 12,
                        "recent_ed_visit": 1,
                        "hba1c": 9.4,
                        "systolic_bp": 165,
                        "egfr": 32,
                        "followup_days": 24,
                    }
                },
            )
        )

        print_result(
            "RISK MCP TOOL RESULT",
            risk_result,
        )

        # ------------------------------------------------
        # Tool 4:
        # Clinical NLP
        # ------------------------------------------------

        nlp_result = (
            await client.call_tool(
                "analyze_clinical_note",
                {
                    "note": (
                        "Synthetic clinical note. "
                        "Patient has chronic "
                        "obstructive pulmonary disease "
                        "with shortness of breath, "
                        "oxygen saturation of 93 "
                        "percent, chronic respiratory "
                        "symptoms, and documented use "
                        "of an albuterol inhaler."
                    ),
                    "top_k": 3,
                },
            )
        )

        print_result(
            "CLINICAL NLP MCP TOOL RESULT",
            nlp_result,
        )


if __name__ == "__main__":

    asyncio.run(
        main()
    )