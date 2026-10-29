"""Workspace domain profiles remain separate from agent responsibilities."""

import json
import re
import unittest
from unittest.mock import patch

from pydantic_ai.models.function import FunctionModel
from test_support import ask
from test_tenants import answering_model

from agent import INSTRUCTIONS, classify_question
from datasource_mapping import TENANTS
from domain_instructions import DOMAIN_PROFILES, DomainProfile, domain_context
from planning import CLASSIFIER_INSTRUCTIONS, PLANNER_INSTRUCTIONS


class DomainInstructionTests(unittest.IsolatedAsyncioTestCase):
    async def test_base_prompts_have_no_workspace_business_rules(self):
        for prompt in (CLASSIFIER_INSTRUCTIONS, PLANNER_INSTRUCTIONS, INSTRUCTIONS):
            for domain_term in (
                "AMC",
                "holdings",
                "NAV",
                "mutual fund",
                "redemption",
                "tax cost basis",
                "SIP",
                "credit card",
            ):
                with self.subTest(domain_term=domain_term):
                    self.assertNotRegex(
                        prompt, r"(?i)\b" + re.escape(domain_term) + r"\b"
                    )

    async def test_stage_and_workspace_isolation(self):
        classifier = json.loads(domain_context("2", tenant_id="2", stage="classifier"))
        planner = json.loads(domain_context("2", tenant_id="2", stage="planner"))
        executor = json.loads(domain_context("2", tenant_id="2", stage="executor"))
        profile = DOMAIN_PROFILES["smartwealth"]
        self.assertEqual(
            classifier["instructions"],
            profile.common_instructions + profile.classifier_instructions,
        )
        self.assertEqual(
            planner["instructions"],
            profile.common_instructions + profile.planner_instructions,
        )
        self.assertEqual(
            executor["instructions"],
            profile.common_instructions + profile.executor_instructions,
        )
        for stage in ("classifier", "planner", "executor"):
            credit = domain_context("1", tenant_id="1", stage=stage)
            self.assertNotIn("Asset Management Company", credit)
            self.assertNotIn("SmartWealth", credit)
        with self.assertRaises(ValueError):
            domain_context("2", tenant_id="1", stage="planner")

    async def test_runtime_profile_injected_into_each_agent_stage(self):
        tenant = TENANTS["1"].model_copy(deep=True)
        tenant.workspaces["1"].domain_profile = "custom_domain"
        profile = DomainProfile(
            description="Custom workspace domain",
            glossary={"ABC": "A custom domain term"},
            common_instructions=["COMMON_MARKER"],
            classifier_instructions=["CLASSIFY_MARKER"],
            planner_instructions=["PLAN_MARKER"],
            executor_instructions=["EXECUTE_MARKER"],
        )
        events = []
        base = answering_model("1", "1", "cc_customers", 20, events)

        def model(messages, info):
            self.assertIn("COMMON_MARKER", info.instructions)
            self.assertIn("A custom domain term", info.instructions)
            stage = "executor"
            if info.output_tools:
                properties = info.output_tools[0].parameters_json_schema["properties"]
                stage = "classifier" if "objective" in properties else "planner"
            markers = {
                "classifier": "CLASSIFY_MARKER",
                "planner": "PLAN_MARKER",
                "executor": "EXECUTE_MARKER",
            }
            self.assertIn(markers[stage], info.instructions)
            for other_stage, marker in markers.items():
                if other_stage != stage:
                    self.assertNotIn(marker, info.instructions)
            return base.function(messages, info)

        with (
            patch.dict(TENANTS, {"1": tenant}),
            patch.dict(DOMAIN_PROFILES, {"custom_domain": profile}),
        ):
            answer = await ask(
                "How many customers?",
                tenant_id="1",
                workspace_id="1",
                model=FunctionModel(model),
            )
        self.assertEqual(answer, "20 customers.")
        self.assertEqual(events, ["classify", "plan", "schema", "sql", "answer"])

    async def test_unknown_profile_fails_before_model_or_sources(self):
        tenant = TENANTS["1"].model_copy(deep=True)
        tenant.workspaces["1"].domain_profile = "does-not-exist"
        with (
            patch.dict(TENANTS, {"1": tenant}),
            patch("pipeline.configured_model") as model,
        ):
            with self.assertRaisesRegex(ValueError, "unknown domain profile"):
                await classify_question(
                    "How many customers?", tenant_id="1", workspace_id="1"
                )
            model.assert_not_called()

    async def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):
            domain_context("1", tenant_id="1", stage="unknown")
