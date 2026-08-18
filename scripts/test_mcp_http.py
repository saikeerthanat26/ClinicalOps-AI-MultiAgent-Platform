import asyncio


from mcp import Client


MCP_URL = (
    "http://127.0.0.1:8000/mcp"
)


async def main() -> None:

    print()
    print(
        "ClinicalOps MCP HTTP Test"
    )

    print(
        "=" * 60
    )

    print(
        f"Connecting to: {MCP_URL}"
    )

    # -----------------------------------------------------
    # Connect through real Streamable HTTP
    # -----------------------------------------------------

    async with Client(
        MCP_URL
    ) as client:

        # -------------------------------------------------
        # Discover MCP tools
        # -------------------------------------------------

        tool_list = (
            await client.list_tools()
        )

        print()
        print(
            "Discovered MCP tools:"
        )

        for tool in tool_list.tools:

            print(
                f"- {tool.name}"
            )

        # -------------------------------------------------
        # Call FHIR tool through HTTP MCP transport
        # -------------------------------------------------

        result = (
            await client.call_tool(
                "get_patient_context",
                {
                    "patient_id": "P001",
                },
            )
        )

        print()
        print(
            "=" * 60
        )

        print(
            "FHIR TOOL OVER HTTP"
        )

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


if __name__ == "__main__":

    asyncio.run(
        main()
    )