"""Tenant ownership, model selection and multiple workspace integration tests."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, structured_response, tool_returns
from test_support import ask

from agent import configured_model, plan_question
from datasource_mapping import (MAPPING, TENANTS, TenantMapping, resolve_scope,
                                routing_catalog)
from planning import ExecutionPlan


def answering_model(tenant_id, workspace_id, table, count, events):
    def model(messages, info):
        if info.output_tools:
            properties = info.output_tools[0].parameters_json_schema["properties"]
            if "objective" in properties:
                events.append("classify")
                return structured_response(
                    info, {"intent": "aggregation", "objective": "Count customers"}
                )
            events.append("plan")
            payload = executable_plan("1" if table.startswith("cc_") else "2")
            payload.update(tenant_id=tenant_id, workspace_id=workspace_id)
            return structured_response(info, payload)
        returns = tool_returns(messages)
        if not returns:
            events.append("schema")
            return ModelResponse(
                parts=[ToolCallPart("describe_sources", {"step_id": "s1"})]
            )
        if returns[-1].tool_name == "describe_sources":
            events.append("sql")
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "run_sql",
                        {
                            "step_id": "s2",
                            "statement": f"SELECT count(*) AS n FROM {table}",
                        },
                    )
                ]
            )
        events.append("answer")
        return ModelResponse(parts=[TextPart(f"{count} customers.")])

    return FunctionModel(model)


class TenantTests(unittest.IsolatedAsyncioTestCase):
    async def test_ids_and_ownership_fail_before_model_or_source(self):
        for tenant_id, workspace_id in [
            ("missing", "1"),
            ("1", "2"),
            ("2", "1"),
            ("1", "missing"),
        ]:
            with self.subTest(tenant_id=tenant_id, workspace_id=workspace_id):
                with (
                    patch("pipeline.configured_model") as model,
                    patch("data_access.create_context") as loader,
                ):
                    with self.assertRaises(ValueError):
                        await ask(
                            "How many customers?",
                            tenant_id=tenant_id,
                            workspace_id=workspace_id,
                        )
                    with self.assertRaises(ValueError):
                        await plan_question(
                            "How many customers?",
                            tenant_id=tenant_id,
                            workspace_id=workspace_id,
                        )
                    model.assert_not_called()
                    loader.assert_not_called()
        with self.assertRaises(TypeError):
            await ask("How many customers?", workspace_id="1")
        with self.assertRaises(TypeError):
            await ask("How many customers?", tenant_id="1")

    async def test_model_mapping_ignores_global_model_override(self):
        with patch.dict(
            os.environ,
            {"LITELLM_MASTER_KEY": "local-test", "LITELLM_MODEL": "wrong-model"},
        ):
            self.assertEqual(configured_model("1", "1").model_name, "qwen-coder")
            self.assertEqual(configured_model("2", "2").model_name, "ornith-reasoner")

    async def test_mapped_model_used_for_all_three_stages(self):
        for tenant_id, workspace_id, model_name, table, count in [
            ("1", "1", "qwen-coder", "cc_customers", 20),
            ("2", "2", "ornith-reasoner", "sw_customers", 12),
        ]:
            events = []
            model = answering_model(tenant_id, workspace_id, table, count, events)
            with (
                self.subTest(tenant_id=tenant_id),
                patch.dict(os.environ, {"LITELLM_MASTER_KEY": "local-test"}),
            ):
                with patch(
                    "model_config.OpenAIChatModel", return_value=model
                ) as factory:
                    answer = await ask(
                        "How many customers?",
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                    )
                self.assertEqual(answer, f"{count} customers.")
                factory.assert_called_once()
                self.assertEqual(factory.call_args.args[0], model_name)
                self.assertEqual(
                    events, ["classify", "plan", "schema", "sql", "answer"]
                )

    async def test_multiple_workspaces_with_distinct_source_locations(self):
        tenant = TENANTS["1"].model_copy(deep=True)
        workspace = tenant.workspaces["1"].model_copy(deep=True)
        with tempfile.TemporaryDirectory() as directory:
            pq.write_table(
                pa.table({"customer_id": ["OTHER1", "OTHER2"]}),
                Path(directory) / "customers.parquet",
            )
            source = workspace.sources["credit_cards"]
            source.location = directory
            source.tables = {"cc_customers": "customers.parquet"}
            tenant.workspaces["additional-workspace"] = workspace
            with patch.dict(TENANTS, {"1": tenant}):
                self.assertEqual(
                    resolve_scope("1", "additional-workspace").model.model_name,
                    "qwen-coder",
                )
                events = []
                answer = await ask(
                    "How many customers?",
                    tenant_id="1",
                    workspace_id="additional-workspace",
                    model=answering_model(
                        "1", "additional-workspace", "cc_customers", 2, events
                    ),
                )
                self.assertEqual(answer, "2 customers.")
                self.assertEqual(events[-1], "answer")
                with self.assertRaises(ValueError):
                    resolve_scope("2", "additional-workspace")

    async def test_workspace_ids_are_resolved_within_tenant(self):
        tenant = TENANTS["2"].model_copy(deep=True)
        tenant.workspaces["1"] = tenant.workspaces["2"].model_copy(deep=True)
        with patch.dict(TENANTS, {"2": tenant}):
            self.assertEqual(resolve_scope("1", "1").workspace.name, "Credit Cards")
            self.assertEqual(resolve_scope("2", "1").workspace.name, "SmartWealth")
            self.assertNotIn("cc_customers", routing_catalog("1", tenant_id="2"))
            plan = executable_plan("2")
            plan["workspace_id"] = "1"
            self.assertEqual(ExecutionPlan.model_validate(plan).tenant_id, "2")

    async def test_mapping_rejects_missing_model_reference(self):
        mapping = MAPPING.model_dump()
        mapping["tenants"]["1"]["model_id"] = "not-configured"
        with self.assertRaises(ValueError):
            TenantMapping.model_validate(mapping)
