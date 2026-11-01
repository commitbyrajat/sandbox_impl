"""Deterministic, delayed pipeline for browser tests only; never used by api:app."""

import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from api import create_app
from execution import ExecutionResult
from pipeline import Pipeline
from planning import QuestionIntent
from runtime import trace
from test_pipeline import adapters


def fixture_pipeline():
    services = adapters([])
    services.model_factory = lambda *args: object()
    services.validate = lambda *args: None
    build = services.plan

    async def classify(question, **kwargs):
        await asyncio.sleep(0.25)
        return QuestionIntent(
            intent="aggregation",
            objective=question,
            constraints=["Use the selected workspace", "Use complete source data"],
        )

    async def plan(*args, **kwargs):
        await asyncio.sleep(0.25)
        return await build(*args, **kwargs)

    async def execute(question, *args):
        trace("SQL_STAGE 1/2: start; chars=240")
        await asyncio.sleep(20 if "slow" in question.lower() else 0.3)
        if "fail" in question.lower():
            raise ValueError(
                "The selected period could not be validated. No answer was produced."
            )
        trace("SQL_STAGE 1/2: completed; rows=20")
        trace("SQL_STAGE 2/2: completed; rows=1")
        return ExecutionResult(
            "Analysis completed.",
            {
                "query_1": {
                    "columns": ["customers"],
                    "rows": [{"customers": 20}],
                    "column_types": [
                        {"name": "customers", "type": "int64", "nullable": False}
                    ],
                }
            },
            ["SELECT COUNT(*) AS customers FROM cc_customers"],
            [],
            ["s1", "s2"],
        )

    async def summarize(*args):
        await asyncio.sleep(0.3)
        return "## Portfolio snapshot\n\nYour workspace contains **20 registered customers**.\n\n| Metric | Value |\n| --- | ---: |\n| Registered customers | 20 |\n\nThe complete registered population was included in this analysis."

    services.classify, services.plan, services.execute, services.summarize = (
        classify,
        plan,
        execute,
        summarize,
    )
    return Pipeline(services)


if __name__ == "__main__":
    uvicorn.run(
        create_app(fixture_pipeline), host="127.0.0.1", port=8012, log_level="warning"
    )
