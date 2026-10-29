"""README business questions must reject SQL-only plans before publication."""

import json
import re
import shlex
import unittest
from pathlib import Path

from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, structured_response

from domain_instructions import get_domain_profile
from planning import QuestionIntent, build_plan


class BusinessHybridCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_documented_business_questions_require_python(self):
        text = (
            Path("README.md")
            .read_text()
            .split("### Changes to the question that would justify SQL + Python", 1)[1]
            .split("To inspect the actual planner decision", 1)[0]
        )
        commands = re.findall(r"`(uv run agent.py[^`]+)`", text)
        self.assertEqual(len(commands), 4)
        profile = get_domain_profile("1", tenant_id="1")
        for command in commands:
            question = shlex.split(command)[-1]
            with self.subTest(question=question):
                self.assertLess(len(question.split()), 30)
                self.assertNotRegex(
                    question.lower(),
                    r"\b(?:python|sql|bootstrap|k-means|seed|iterations)\b",
                )
                self.assertTrue(
                    any(
                        re.search(pattern, question, re.I)
                        for pattern in profile.python_required_patterns
                    )
                )
                calls = []

                def planner(messages, info):
                    calls.append(1)
                    return structured_response(
                        info,
                        executable_plan(
                            "1", "sql" if len(calls) == 1 else "sql_python"
                        ),
                    )

                plan = await build_plan(
                    question,
                    QuestionIntent(intent="advanced_analysis", objective=question),
                    tenant_id="1",
                    workspace_id="1",
                    model=FunctionModel(planner),
                )
                self.assertEqual(len(calls), 2)
                self.assertEqual(plan.execution_mode, "sql_python")
                self.assertTrue(any(s.action == "python" for s in plan.steps))

    async def test_ordinary_financial_questions_do_not_match_required_python_patterns(
        self,
    ):
        profile = get_domain_profile("1", tenant_id="1")
        for line in Path("QUES.md").read_text().splitlines():
            self.assertFalse(
                any(
                    re.search(pattern, line, re.I)
                    for pattern in profile.python_required_patterns
                )
            )
