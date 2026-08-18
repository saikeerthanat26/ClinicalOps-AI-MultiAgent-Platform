from typing import Any

from mcp import Client
from mcp.types import TextContent


MCP_URL = (
    "http://127.0.0.1:8000/mcp"
)


class ClinicalOpsMCPClient:

    def __init__(
        self,
        url: str = MCP_URL,
    ) -> None:

        self.url = url


    # -----------------------------------------------------
    # Generic MCP tool caller
    # -----------------------------------------------------

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        async with Client(
            self.url
        ) as client:

            result = (
                await client.call_tool(
                    tool_name,
                    arguments,
                )
            )

        # -------------------------------------------------
        # MCP tool-level error
        # -------------------------------------------------

        if result.is_error:

            error_messages = []

            for block in result.content:

                if isinstance(
                    block,
                    TextContent,
                ):

                    error_messages.append(
                        block.text
                    )

            error_text = (
                " | ".join(
                    error_messages
                )
                or (
                    "Unknown MCP tool "
                    "execution error."
                )
            )

            raise RuntimeError(
                f"MCP tool "
                f"{tool_name} failed: "
                f"{error_text}"
            )

        # -------------------------------------------------
        # Structured MCP result
        # -------------------------------------------------

        if (
            result.structured_content
            is None
        ):

            raise RuntimeError(
                f"MCP tool {tool_name} "
                "returned no structured content."
            )

        return dict(
            result.structured_content
        )


    # -----------------------------------------------------
    # FHIR MCP tool
    # -----------------------------------------------------

    async def get_patient_context(
        self,
        patient_id: str,
    ) -> dict[str, Any]:

        return await self.call_tool(
            "get_patient_context",
            {
                "patient_id": (
                    patient_id
                ),
            },
        )


    # -----------------------------------------------------
    # RAG MCP tool
    # -----------------------------------------------------

    async def search_healthcare_knowledge(
        self,
        query: str,
        top_k: int = 3,
    ) -> dict[str, Any]:

        return await self.call_tool(
            "search_healthcare_knowledge",
            {
                "query": query,
                "top_k": top_k,
            },
        )


    # -----------------------------------------------------
    # Risk MCP tool
    # -----------------------------------------------------

    async def predict_readmission_risk(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:

        return await self.call_tool(
            "predict_readmission_risk",
            {
                "features": features,
            },
        )


    # -----------------------------------------------------
    # NLP MCP tool
    # -----------------------------------------------------

    async def analyze_clinical_note(
        self,
        note: str,
        top_k: int = 3,
    ) -> dict[str, Any]:

        return await self.call_tool(
            "analyze_clinical_note",
            {
                "note": note,
                "top_k": top_k,
            },
        )


clinicalops_mcp_client = (
    ClinicalOpsMCPClient()
)