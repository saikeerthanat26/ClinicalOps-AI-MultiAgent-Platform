import asyncio
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


EVAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "agent_eval_cases.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase11_evaluation.json"
)

AGENT_URL = (
    "http://127.0.0.1:8000"
    "/api/v1/agents/run"
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def percentage(
    passed: int,
    total: int,
) -> float:

    if total == 0:
        return 0.0

    return round(
        (
            passed
            / total
        )
        * 100,
        2,
    )


def first_source_id(
    response: dict[str, Any],
) -> str | None:

    sources = (
        response
        .get(
            "data",
            {},
        )
        .get(
            "sources",
            [],
        )
    )

    if not sources:
        return None

    return sources[
        0
    ].get(
        "id"
    )


def first_pattern_id(
    response: dict[str, Any],
) -> str | None:

    matches = (
        response
        .get(
            "data",
            {},
        )
        .get(
            "matches",
            [],
        )
    )

    if not matches:
        return None

    return matches[
        0
    ].get(
        "id"
    )


# ---------------------------------------------------------
# Evaluate one response
# ---------------------------------------------------------

def evaluate_response(
    response: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, bool]:

    checks: dict[
        str,
        bool,
    ] = {}


    # -----------------------------------------------------
    # Route
    # -----------------------------------------------------

    if "route" in expected:

        checks[
            "route"
        ] = (
            response.get(
                "route"
            )
            == expected[
                "route"
            ]
        )


    # -----------------------------------------------------
    # Agent
    # -----------------------------------------------------

    if "agent_used" in expected:

        checks[
            "agent"
        ] = (
            response.get(
                "agent_used"
            )
            == expected[
                "agent_used"
            ]
        )


    # -----------------------------------------------------
    # Guardrail decision
    # -----------------------------------------------------

    if "guardrail_blocked" in expected:

        checks[
            "guardrail_decision"
        ] = (
            response.get(
                "guardrail_blocked"
            )
            == expected[
                "guardrail_blocked"
            ]
        )


    if (
        "input_guardrail_passed"
        in expected
    ):

        checks[
            "input_guardrail"
        ] = (
            response.get(
                "input_guardrail_passed"
            )
            == expected[
                "input_guardrail_passed"
            ]
        )


    if (
        "output_guardrail_passed"
        in expected
    ):

        checks[
            "output_guardrail"
        ] = (
            response.get(
                "output_guardrail_passed"
            )
            == expected[
                "output_guardrail_passed"
            ]
        )


    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    if "verified" in expected:

        checks[
            "verifier"
        ] = (
            response.get(
                "verified"
            )
            == expected[
                "verified"
            ]
        )


    # -----------------------------------------------------
    # MCP tool selection
    # -----------------------------------------------------

    if "mcp_tool" in expected:

        actual_mcp_tool = (
            response
            .get(
                "data",
                {},
            )
            .get(
                "mcp_tool"
            )
        )

        checks[
            "mcp_tool"
        ] = (
            actual_mcp_tool
            == expected[
                "mcp_tool"
            ]
        )


    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------

    if (
        "retrieval_relevant"
        in expected
    ):

        actual_relevant = (
            response
            .get(
                "data",
                {},
            )
            .get(
                "retrieval_relevant"
            )
        )

        checks[
            "retrieval_relevance"
        ] = (
            actual_relevant
            == expected[
                "retrieval_relevant"
            ]
        )


    if "top_source_id" in expected:

        checks[
            "rag_top1"
        ] = (
            first_source_id(
                response
            )
            == expected[
                "top_source_id"
            ]
        )


    # -----------------------------------------------------
    # Clinical NLP
    # -----------------------------------------------------

    if "top_pattern_id" in expected:

        checks[
            "nlp_top1"
        ] = (
            first_pattern_id(
                response
            )
            == expected[
                "top_pattern_id"
            ]
        )


    # -----------------------------------------------------
    # Risk
    # -----------------------------------------------------

    if (
        "predicted_readmission"
        in expected
    ):

        actual_prediction = (
            response
            .get(
                "data",
                {},
            )
            .get(
                "predicted_readmission"
            )
        )

        checks[
            "risk_classification"
        ] = (
            actual_prediction
            == expected[
                "predicted_readmission"
            ]
        )


    # -----------------------------------------------------
    # Required guardrail flags
    # -----------------------------------------------------

    if (
        "required_input_flag"
        in expected
    ):

        actual_flags = (
            response.get(
                "input_guardrail_flags",
                [],
            )
        )

        checks[
            "required_input_flag"
        ] = (
            expected[
                "required_input_flag"
            ]
            in actual_flags
        )


    if (
        "required_input_flags"
        in expected
    ):

        actual_flags = set(
            response.get(
                "input_guardrail_flags",
                [],
            )
        )

        required_flags = set(
            expected[
                "required_input_flags"
            ]
        )

        checks[
            "required_input_flags"
        ] = (
            required_flags
            .issubset(
                actual_flags
            )
        )


    if (
        "required_output_flag"
        in expected
    ):

        actual_flags = (
            response.get(
                "output_guardrail_flags",
                [],
            )
        )

        checks[
            "required_output_flag"
        ] = (
            expected[
                "required_output_flag"
            ]
            in actual_flags
        )


    return checks


# ---------------------------------------------------------
# Run one case
# ---------------------------------------------------------

async def run_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
) -> dict[str, Any]:

    payload = deepcopy(
        case[
            "request"
        ]
    )

    payload[
        "thread_id"
    ] = (
        "eval-"
        f"{case['id']}-"
        f"{uuid4()}"
    )

    start = (
        time.perf_counter()
    )

    try:

        response = (
            await client.post(
                AGENT_URL,
                json=payload,
            )
        )

        latency_ms = round(
            (
                time.perf_counter()
                - start
            )
            * 1000,
            2,
        )

        http_success = (
            response.status_code
            == 200
        )

        if not http_success:

            return {
                "id": case[
                    "id"
                ],
                "name": case[
                    "name"
                ],
                "http_success": False,
                "status_code": (
                    response.status_code
                ),
                "latency_ms": (
                    latency_ms
                ),
                "checks": {},
                "passed": False,
                "error": (
                    response.text
                ),
            }

        body = (
            response.json()
        )

        checks = (
            evaluate_response(
                body,
                case[
                    "expected"
                ],
            )
        )

        passed = (
            all(
                checks.values()
            )
            if checks
            else False
        )

        return {
            "id": case[
                "id"
            ],
            "name": case[
                "name"
            ],
            "http_success": True,
            "status_code": 200,
            "latency_ms": (
                latency_ms
            ),
            "route": body.get(
                "route"
            ),
            "agent_used": body.get(
                "agent_used"
            ),
            "checks": checks,
            "passed": passed,
        }

    except Exception as error:

        latency_ms = round(
            (
                time.perf_counter()
                - start
            )
            * 1000,
            2,
        )

        return {
            "id": case[
                "id"
            ],
            "name": case[
                "name"
            ],
            "http_success": False,
            "status_code": None,
            "latency_ms": (
                latency_ms
            ),
            "checks": {},
            "passed": False,
            "error": str(
                error
            ),
        }


# ---------------------------------------------------------
# Aggregate metric
# ---------------------------------------------------------

def aggregate_check(
    results: list[dict[str, Any]],
    check_name: str,
) -> dict[str, Any]:

    applicable = []

    for result in results:

        checks = result.get(
            "checks",
            {},
        )

        if check_name in checks:

            applicable.append(
                checks[
                    check_name
                ]
            )

    passed = sum(
        1
        for value in applicable
        if value
    )

    total = len(
        applicable
    )

    return {
        "passed": passed,
        "total": total,
        "accuracy_percent": (
            percentage(
                passed,
                total,
            )
        ),
    }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

async def main() -> None:

    with EVAL_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        cases = json.load(
            file
        )


    print()
    print(
        "ClinicalOps AI - Phase 11 Evaluation"
    )

    print(
        "=" * 70
    )

    print(
        f"Evaluation cases: {len(cases)}"
    )

    print(
        f"Endpoint: {AGENT_URL}"
    )

    print()


    results = []


    async with httpx.AsyncClient(
        timeout=180.0
    ) as client:

        # Run sequentially because local models
        # share CPU/MPS resources.

        for (
            index,
            case,
        ) in enumerate(
            cases,
            start=1,
        ):

            print(
                f"[{index}/{len(cases)}] "
                f"{case['id']} - "
                f"{case['name']}"
            )

            result = (
                await run_case(
                    client,
                    case,
                )
            )

            results.append(
                result
            )

            status = (
                "PASS"
                if result[
                    "passed"
                ]
                else "FAIL"
            )

            print(
                f"    Result: {status}"
            )

            print(
                "    Latency: "
                f"{result['latency_ms']} ms"
            )

            failed_checks = [
                name
                for (
                    name,
                    passed,
                )
                in result.get(
                    "checks",
                    {},
                ).items()
                if not passed
            ]

            if failed_checks:

                print(
                    "    Failed checks: "
                    + ", ".join(
                        failed_checks
                    )
                )

            if result.get(
                "error"
            ):

                print(
                    "    Error: "
                    f"{result['error']}"
                )

            print()


    # -----------------------------------------------------
    # Core metrics
    # -----------------------------------------------------

    total_cases = len(
        results
    )

    passed_cases = sum(
        1
        for result in results
        if result[
            "passed"
        ]
    )

    successful_http = sum(
        1
        for result in results
        if result[
            "http_success"
        ]
    )


    metric_names = [
        "route",
        "agent",
        "guardrail_decision",
        "input_guardrail",
        "output_guardrail",
        "verifier",
        "mcp_tool",
        "retrieval_relevance",
        "rag_top1",
        "nlp_top1",
        "risk_classification",
        "required_input_flag",
        "required_input_flags",
        "required_output_flag",
    ]


    metrics = {
        name: aggregate_check(
            results,
            name,
        )
        for name in metric_names
    }


    average_latency = round(
        sum(
            result[
                "latency_ms"
            ]
            for result in results
        )
        / total_cases,
        2,
    )


    report = {
        "evaluation": (
            "ClinicalOps AI "
            "Phase 11 Evaluation"
        ),

        "generated_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),

        "total_cases": (
            total_cases
        ),

        "passed_cases": (
            passed_cases
        ),

        "failed_cases": (
            total_cases
            - passed_cases
        ),

        "http_success_rate_percent": (
            percentage(
                successful_http,
                total_cases,
            )
        ),

        "end_to_end_pass_rate_percent": (
            percentage(
                passed_cases,
                total_cases,
            )
        ),

        "average_latency_ms": (
            average_latency
        ),

        "metrics": (
            metrics
        ),

        "results": (
            results
        ),
    }


    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )


    # -----------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )


    print(
        "HTTP success rate: "
        f"{report['http_success_rate_percent']}%"
    )

    print(
        "End-to-end pass rate: "
        f"{report['end_to_end_pass_rate_percent']}%"
    )

    print(
        "Average latency: "
        f"{average_latency} ms"
    )

    print()


    display_metrics = [
        (
            "Routing accuracy",
            "route",
        ),
        (
            "Agent selection accuracy",
            "agent",
        ),
        (
            "Guardrail decision accuracy",
            "guardrail_decision",
        ),
        (
            "Input guardrail accuracy",
            "input_guardrail",
        ),
        (
            "Output guardrail accuracy",
            "output_guardrail",
        ),
        (
            "Verifier accuracy",
            "verifier",
        ),
        (
            "MCP tool accuracy",
            "mcp_tool",
        ),
        (
            "Retrieval relevance accuracy",
            "retrieval_relevance",
        ),
        (
            "RAG Top-1 accuracy",
            "rag_top1",
        ),
        (
            "Clinical NLP Top-1 accuracy",
            "nlp_top1",
        ),
        (
            "Risk classification accuracy",
            "risk_classification",
        ),
    ]


    for (
        label,
        metric_name,
    ) in display_metrics:

        metric = metrics[
            metric_name
        ]

        if metric[
            "total"
        ] == 0:

            continue

        print(
            f"{label}: "
            f"{metric['accuracy_percent']}% "
            f"({metric['passed']}/"
            f"{metric['total']})"
        )


    print()

    print(
        "Report saved to:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":

    asyncio.run(
        main()
    )