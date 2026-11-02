"""Deterministic, delayed pipeline for browser tests only; never used by api:app."""

import asyncio
from decimal import Decimal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn
from pydantic_monty import AsyncMonty
from api import create_app
from execution import ExecutionResult
from pipeline import Pipeline
from planning import QuestionIntent
from runtime import trace
from test_pipeline import adapters

PYTHON_SOURCE = """from decimal import Decimal

# Keep SQL amounts exact; values are in ₹.
rows = data["query_1"]
amounts = [row["amount_inr"] for row in rows]
average = sum(amounts, Decimal("0")) / Decimal(len(amounts))

result = {"average_inr": str(average), "row_count": len(rows)}
result
"""

SQL_SOURCES = [
    "-- Customer population\nSELECT customer_id\nFROM cc_customers\nORDER BY customer_id;\n",
    "SELECT card_id, SUM(approved_net_spend_inr) AS spending_inr\nFROM cc_monthly_financials\nGROUP BY card_id;\n",
    "-- Monetary inputs – ₹\nSELECT profit_inr AS amount_inr\nFROM cc_monthly_financials\nORDER BY card_id;\n",
]


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
        if "python" in question.lower():
            rows = [{"amount_inr": Decimal("12.30")}, {"amount_inr": Decimal("14.20")}]
            async with AsyncMonty(
                binary_path=str(Path(sys.executable).parent / "monty"),
                min_processes=1,
                max_processes=1,
            ) as pool:
                async with pool.checkout() as session:
                    output = await session.feed_run(
                        PYTHON_SOURCE, inputs={"data": {"query_1": rows}}
                    )
            runs = [
                {
                    "step_id": "py3",
                    "executed": True,
                    "code": PYTHON_SOURCE,
                    "result_ids": ["query_1"],
                    "output": output,
                }
            ]
            if "reuse" in question.lower():
                runs.append(
                    {
                        "step_id": "py4",
                        "executed": False,
                        "reused_from_step_id": "py3",
                        "result_ids": ["query_1"],
                        "output": output,
                    }
                )
            return ExecutionResult(
                "Analysis completed.",
                {"query_1": {"columns": ["amount_inr"], "rows": rows}},
                SQL_SOURCES,
                runs,
                ["s1", "s2", "py3"],
            )
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
