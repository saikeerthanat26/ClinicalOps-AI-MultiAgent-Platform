import json
import logging
import math

from collections import Counter
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

LOG_FILE = (
    LOG_DIR
    / "agent_requests.jsonl"
)


LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Dedicated ClinicalOps observability logger
# ---------------------------------------------------------

logger = logging.getLogger(
    "clinicalops.agent_observability"
)

logger.setLevel(
    logging.INFO
)

logger.propagate = False


if not logger.handlers:

    file_handler = (
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        )
    )

    file_handler.setFormatter(
        logging.Formatter(
            "%(message)s"
        )
    )

    logger.addHandler(
        file_handler
    )


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _percentage(
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


def _percentile(
    values: list[float],
    percentile: float,
) -> float:

    if not values:
        return 0.0

    ordered = sorted(
        values
    )

    position = (
        math.ceil(
            percentile
            * len(
                ordered
            )
        )
        - 1
    )

    position = max(
        0,
        min(
            position,
            len(
                ordered
            )
            - 1,
        ),
    )

    return round(
        ordered[
            position
        ],
        2,
    )


# ---------------------------------------------------------
# Observability service
# ---------------------------------------------------------

class ClinicalOpsObservabilityService:

    def record(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any]:

        payload = {
            "timestamp_utc": (
                datetime.now(
                    timezone.utc
                )
                .isoformat()
            ),
            **event,
        }

        logger.info(
            json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
        )

        return payload


    def read_events(
        self,
    ) -> list[dict[str, Any]]:

        if not LOG_FILE.exists():

            return []

        events = []

        with LOG_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    events.append(
                        json.loads(
                            line
                        )
                    )

                except json.JSONDecodeError:

                    continue

        return events


    def recent(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:

        events = (
            self.read_events()
        )

        return events[
            -limit:
        ][::-1]


    def summary(
        self,
    ) -> dict[str, Any]:

        events = (
            self.read_events()
        )

        if not events:

            return {
                "total_requests": 0,
                "message": (
                    "No observability "
                    "events recorded yet."
                ),
                "log_file": str(
                    LOG_FILE
                ),
            }


        # -------------------------------------------------
        # Latency
        # -------------------------------------------------

        latencies = [
            float(
                event[
                    "latency_ms"
                ]
            )
            for event in events
            if event.get(
                "latency_ms"
            )
            is not None
        ]


        average_latency = (
            round(
                sum(
                    latencies
                )
                / len(
                    latencies
                ),
                2,
            )
            if latencies
            else 0.0
        )


        # -------------------------------------------------
        # Counts
        # -------------------------------------------------

        status_counts = Counter(
            event.get(
                "execution_status",
                "unknown",
            )
            for event in events
        )

        route_counts = Counter(
            event.get(
                "route"
            )
            for event in events
            if event.get(
                "route"
            )
        )

        agent_counts = Counter(
            event.get(
                "agent_used"
            )
            for event in events
            if event.get(
                "agent_used"
            )
        )

        mcp_tool_counts = Counter(
            event.get(
                "mcp_tool"
            )
            for event in events
            if event.get(
                "mcp_tool"
            )
        )


        # -------------------------------------------------
        # Guardrails
        # -------------------------------------------------

        blocked_count = sum(
            1
            for event in events
            if event.get(
                "guardrail_blocked"
            )
            is True
        )


        # -------------------------------------------------
        # Verifier accuracy for successfully executed
        # specialist-agent requests.
        # -------------------------------------------------

        verifier_events = [
            event
            for event in events
            if (
                event.get(
                    "execution_status"
                )
                == "success"
                and event.get(
                    "verified"
                )
                is not None
            )
        ]

        verifier_passed = sum(
            1
            for event in verifier_events
            if event.get(
                "verified"
            )
            is True
        )


        # -------------------------------------------------
        # Output guardrail success for specialist requests
        # -------------------------------------------------

        output_guardrail_events = [
            event
            for event in events
            if (
                event.get(
                    "execution_status"
                )
                == "success"
                and event.get(
                    "output_guardrail_passed"
                )
                is not None
            )
        ]

        output_guardrail_passed = sum(
            1
            for event
            in output_guardrail_events
            if event.get(
                "output_guardrail_passed"
            )
            is True
        )


        # -------------------------------------------------
        # Request completion rate
        #
        # A safely blocked guardrail request is still a
        # successfully handled request.
        # -------------------------------------------------

        completed_count = sum(
            1
            for event in events
            if event.get(
                "execution_status"
            )
            in {
                "success",
                "blocked",
            }
        )


        return {
            "total_requests": len(
                events
            ),

            "request_completion_rate_percent": (
                _percentage(
                    completed_count,
                    len(
                        events
                    ),
                )
            ),

            "execution_status_counts": (
                dict(
                    status_counts
                )
            ),

            "route_counts": (
                dict(
                    route_counts
                )
            ),

            "agent_counts": (
                dict(
                    agent_counts
                )
            ),

            "mcp_tool_counts": (
                dict(
                    mcp_tool_counts
                )
            ),

            "guardrail_block_count": (
                blocked_count
            ),

            "guardrail_block_rate_percent": (
                _percentage(
                    blocked_count,
                    len(
                        events
                    ),
                )
            ),

            "verifier_pass_rate_percent": (
                _percentage(
                    verifier_passed,
                    len(
                        verifier_events
                    ),
                )
            ),

            "output_guardrail_pass_rate_percent": (
                _percentage(
                    output_guardrail_passed,
                    len(
                        output_guardrail_events
                    ),
                )
            ),

            "latency_ms": {
                "average": (
                    average_latency
                ),

                "p50": (
                    _percentile(
                        latencies,
                        0.50,
                    )
                ),

                "p95": (
                    _percentile(
                        latencies,
                        0.95,
                    )
                ),

                "max": (
                    round(
                        max(
                            latencies
                        ),
                        2,
                    )
                    if latencies
                    else 0.0
                ),
            },

            "privacy": {
                "questions_logged": False,
                "clinical_notes_logged": False,
                "patient_payloads_logged": False,
                "risk_feature_values_logged": False,
            },

            "log_file": str(
                LOG_FILE
            ),
        }


observability_service = (
    ClinicalOpsObservabilityService()
)