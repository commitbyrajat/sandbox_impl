"""Validate SQL structure, configured join semantics, and the real model retry wire."""

import io
import json
import os
import unittest
from contextlib import redirect_stderr
from datetime import date
from decimal import Decimal
from unittest.mock import patch

import pyarrow as pa
from datafusion import SessionContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from test_agent import executable_plan, tool_returns
from test_model_protocol import completion, mock_provider

from data_access import DataSources
from datasource_mapping import resolve_scope
from domain_instructions import get_domain_profile
from execution import execute_plan
from main import create_context
from model_config import configured_model
from planning import ExecutionPlan, QuestionIntent, build_plan
from runtime import TRACE_ENABLED
from sql_diagnostics import SQLPreflightError, preflight_sql, sql_fingerprint
from sql_joins import JoinBinding, configured_joins, validate_join_metadata


def lab():
    context = SessionContext()
    tables = {
        "accounts": pa.table(
            {"account_id": [1, 2, 3], "name": ["North", "South", "East"]}
        ),
        "ledger": pa.table(
            {
                "record_id": [11, 12, 13],
                "account_id": [1, 1, 2],
                "period": [date(2026, 1, 1), date(2026, 2, 1), date(2026, 1, 1)],
                "amount": pa.array(
                    [Decimal("10"), Decimal("20"), Decimal("30")],
                    type=pa.decimal128(18, 2),
                ),
                "cost": pa.array(
                    [Decimal("1"), Decimal("2"), Decimal("3")],
                    type=pa.decimal128(18, 2),
                ),
            }
        ),
        "items": pa.table({"order_id": [1, 1, 2], "line_no": [1, 2, 1]}),
        "deliveries": pa.table(
            {"order_id": [1, 1, 2], "line_no": [1, 2, 1], "quantity": [4, 5, 6]}
        ),
    }
    for name, table in tables.items():
        context.register_record_batches(name, [table.to_batches()])
    return context, {name: table.schema for name, table in tables.items()}


def sample_bindings():
    return [
        JoinBinding(
            relationship_id="ledger_account",
            left_table="ledger",
            right_table="accounts",
            column_pairs=[("account_id", "account_id")],
            business={
                "meaning": "A ledger observation belongs to its registered account.",
                "left_grain": ["record_id"],
                "right_grain": ["account_id"],
                "cardinality": "many_to_one",
                "aggregation_rules": [
                    "Aggregate observations before combining with another repeated-key fact source."
                ],
            },
        ),
        JoinBinding(
            relationship_id="item_delivery",
            left_table="items",
            right_table="deliveries",
            column_pairs=[("order_id", "order_id"), ("line_no", "line_no")],
            business={
                "meaning": "Deliveries match the exact order line.",
                "left_grain": ["order_id", "line_no"],
                "right_grain": ["order_id", "line_no"],
                "cardinality": "one_to_one",
                "aggregation_rules": [],
            },
        ),
    ]


BAD_CTE = """WITH latest AS (
 SELECT account_id,MAX(period) AS period FROM ledger GROUP BY account_id
) SELECT l.amount,l.cost FROM latest l"""


class SQLPlannerRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_plan_cannot_invent_recipe_and_bypass_preflight(self):
        attempts = 0

        def respond(messages, info):
            nonlocal attempts
            attempts += 1
            plan = executable_plan(declared=True)
            if attempts == 1:
                plan["sql_recipe"] = {
                    "recipe_id": "invented_recipe",
                    "parameters": {},
                    "template_sha256": "invented_hash",
                }
            else:
                feedback = str(messages[-1])
                self.assertIn("sql_recipe", feedback)
                self.assertIn("Input should be None", feedback)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, plan)])

        plan = await build_plan(
            "Count customers",
            QuestionIntent(intent="aggregation", objective="Count customers"),
            tenant_id="1",
            workspace_id="1",
            model=FunctionModel(respond),
        )
        self.assertEqual(attempts, 2)
        self.assertIsNone(plan.sql_recipe)


class SQLPreflightTests(unittest.TestCase):
    def test_unqualified_cte_projection_and_wrong_join_are_reported_together(self):
        context, schemas = lab()
        with self.assertRaises(SQLPreflightError) as raised:
            preflight_sql(
                "WITH observations AS (SELECT l.account_id,l.period,l.amount FROM ledger l JOIN accounts a ON l.record_id=a.account_id) SELECT account_id,latest_period FROM observations",
                schemas,
                context,
                sample_bindings(),
            )
        issues = raised.exception.diagnostic.issues
        missing = next(issue for issue in issues if issue.kind == "unbound_column")
        self.assertEqual(missing.reference, "latest_period")
        self.assertIn("period", missing.available_columns)
        self.assertTrue(any(issue.kind == "invalid_join_keys" for issue in issues))

    def test_dropped_cte_fields_are_reported_together_before_engine_compilation(self):
        context, schemas = lab()
        with patch.object(
            context, "sql", side_effect=AssertionError("must not compile")
        ):
            with self.assertRaises(SQLPreflightError) as raised:
                preflight_sql(BAD_CTE, schemas, context)
        diagnostic = raised.exception.diagnostic
        self.assertEqual(diagnostic.phase, "qualification")
        self.assertTrue(diagnostic.cacheable)
        self.assertEqual(
            {issue.reference for issue in diagnostic.issues}, {"l.amount", "l.cost"}
        )
        self.assertTrue(
            all(issue.kind == "missing_cte_projection" for issue in diagnostic.issues)
        )
        self.assertEqual(
            diagnostic.issues[0].available_columns, ["account_id", "period"]
        )
        self.assertIn("amount", diagnostic.referenced_base_column_types["ledger"])
        self.assertEqual(diagnostic.failed_sql, BAD_CTE)

    def test_corrected_query_is_compiled_unchanged_and_actual_types_are_returned(self):
        context, schemas = lab()
        query = """WITH latest AS (SELECT account_id,MAX(period) AS period FROM ledger GROUP BY account_id)
        SELECT l.amount,l.cost FROM ledger l JOIN latest x
        ON l.account_id=x.account_id AND l.period=x.period"""
        with patch.object(context, "sql", wraps=context.sql) as compile_query:
            check = preflight_sql(query, schemas, context)
        self.assertEqual(compile_query.call_args.args[0], query)
        self.assertTrue(check.datafusion_compiled)
        self.assertEqual(check.result_schema[0]["arrow_type"], "decimal128(18, 2)")
        self.assertEqual(len(context.sql(query).to_pylist()), 2)

    def test_engine_is_authoritative_about_decimal_aggregate_precision(self):
        context, schemas = lab()
        check = preflight_sql(
            "SELECT SUM(amount) AS total FROM ledger", schemas, context
        )
        self.assertEqual(check.result_schema[0]["arrow_type"], "decimal128(28, 2)")
        self.assertEqual(check.inferred_types[0]["type"], "DECIMAL(18, 2)")

    def test_type_inference_failure_is_advisory(self):
        context, schemas = lab()
        with patch(
            "sql_diagnostics.annotate_types",
            side_effect=ValueError("Unknown type rule"),
        ):
            check = preflight_sql("SELECT amount FROM ledger", schemas, context)
        self.assertTrue(check.datafusion_compiled)
        self.assertTrue(check.warnings)
        self.assertEqual(check.inferred_types, [])

    def test_unsupported_function_is_rejected_by_datafusion_preflight(self):
        context, schemas = lab()
        with self.assertRaises(SQLPreflightError) as raised:
            preflight_sql(
                "SELECT no_such_function(amount) FROM ledger", schemas, context
            )
        self.assertEqual(raised.exception.diagnostic.phase, "datafusion_compile")
        self.assertTrue(raised.exception.diagnostic.cacheable)

    def test_scope_and_read_only_restrictions_run_before_engine_calls(self):
        context, schemas = lab()
        for statement in (
            "SELECT * FROM hidden",
            "SELECT * FROM public.accounts",
            "SELECT * FROM read_parquet('/tmp/private.parquet')",
            "DELETE FROM ledger",
            "SELECT 1; SELECT 2",
        ):
            with (
                self.subTest(statement=statement),
                patch.object(
                    context, "sql", side_effect=AssertionError("must not compile")
                ),
            ):
                with self.assertRaises(SQLPreflightError):
                    preflight_sql(statement, schemas, context)

    def test_fingerprint_preserves_literals_and_quoted_identifier_case(self):
        self.assertEqual(
            sql_fingerprint("SELECT amount FROM ledger"),
            sql_fingerprint("-- retry\nselect AMOUNT from LEDGER;"),
        )
        self.assertNotEqual(
            sql_fingerprint("SELECT 'ABC'"), sql_fingerprint("SELECT 'abc'")
        )
        self.assertNotEqual(
            sql_fingerprint('SELECT "Amount" FROM ledger'),
            sql_fingerprint('SELECT "amount" FROM ledger'),
        )


class JoinSemanticsTests(unittest.TestCase):
    def test_business_meaning_and_key_metadata_are_bound_in_both_workspaces(self):
        for tenant, workspace, source in (
            ("1", "1", "credit_cards"),
            ("2", "2", "smartwealth"),
        ):
            with self.subTest(workspace=workspace):
                scope = resolve_scope(tenant, workspace)
                bindings = configured_joins(
                    scope.workspace.sources,
                    get_domain_profile(workspace, tenant_id=tenant).join_semantics,
                )
                context = create_context(source_ids=[source])
                schemas = {
                    table: context.table(table).schema()
                    for table in scope.workspace.sources[source].tables
                }
                validate_join_metadata(bindings, schemas)
                self.assertTrue(bindings)
                self.assertTrue(
                    all(
                        binding.business and binding.business["meaning"]
                        for binding in bindings
                    )
                )

    def test_valid_join_exposes_business_meaning_and_grain_rules(self):
        context, schemas = lab()
        check = preflight_sql(
            "SELECT a.name,SUM(l.amount) AS total FROM ledger l JOIN accounts a ON l.account_id=a.account_id GROUP BY a.name",
            schemas,
            context,
            sample_bindings(),
        )
        reviewed = check.reviewed_joins[0]
        self.assertEqual(reviewed["relationship_id"], "ledger_account")
        self.assertIn("registered account", reviewed["business"]["meaning"])
        self.assertEqual(reviewed["business"]["left_grain"], ["record_id"])
        self.assertEqual(check.warnings[0]["kind"], "aggregation_grain_review")

    def test_wrong_same_type_join_key_is_rejected_before_engine_compilation(self):
        context, schemas = lab()
        query = "SELECT a.name,l.amount FROM ledger l JOIN accounts a ON l.record_id=a.account_id"
        with (
            patch.object(
                context, "sql", side_effect=AssertionError("must not compile")
            ),
            self.assertRaises(SQLPreflightError) as raised,
        ):
            preflight_sql(query, schemas, context, sample_bindings())
        self.assertEqual(raised.exception.diagnostic.phase, "join_keys")
        self.assertEqual(
            raised.exception.diagnostic.issues[0].kind, "invalid_join_keys"
        )
        self.assertIn(
            "registered account", raised.exception.diagnostic.issues[0].repair
        )

    def test_composite_keys_cannot_be_split_across_aliases_or_or_branches(self):
        context, schemas = lab()
        for query in (
            "SELECT i.order_id FROM items i JOIN deliveries d ON i.order_id=d.order_id",
            "SELECT i.order_id FROM items i JOIN deliveries d ON i.order_id=d.order_id JOIN deliveries b ON i.line_no=b.line_no",
            "SELECT i.order_id FROM items i JOIN deliveries d ON i.order_id=d.order_id OR i.line_no=d.line_no",
            "SELECT i.order_id FROM items i JOIN deliveries d USING(order_id)",
        ):
            with self.subTest(query=query), self.assertRaises(SQLPreflightError):
                preflight_sql(query, schemas, context, sample_bindings())
        check = preflight_sql(
            "SELECT i.order_id FROM items i JOIN deliveries d ON i.order_id=d.order_id AND i.line_no=d.line_no",
            schemas,
            context,
            sample_bindings(),
        )
        self.assertTrue(check.datafusion_compiled)
        self.assertTrue(
            preflight_sql(
                "SELECT i.order_id FROM items i JOIN deliveries d USING(order_id,line_no)",
                schemas,
                context,
                sample_bindings(),
            ).datafusion_compiled
        )

    def test_grouped_ctes_can_join_at_a_proven_coarser_grain(self):
        context, schemas = lab()
        query = """WITH i AS (SELECT order_id,COUNT(*) AS n FROM items GROUP BY order_id),
        d AS (SELECT order_id,SUM(quantity) AS quantity FROM deliveries GROUP BY order_id)
        SELECT i.order_id,d.quantity FROM i JOIN d ON i.order_id=d.order_id"""
        check = preflight_sql(query, schemas, context, sample_bindings())
        self.assertEqual(
            check.reviewed_joins[0]["verification_basis"], "unique_aggregate_grain"
        )

    def test_computed_join_lineage_is_not_claimed_as_verified(self):
        context, schemas = lab()
        check = preflight_sql(
            "SELECT a.name FROM ledger l JOIN accounts a ON CAST(l.account_id AS BIGINT)=a.account_id",
            schemas,
            context,
            sample_bindings(),
        )
        self.assertFalse(check.all_join_keys_resolved)
        self.assertTrue(
            any(
                warning.get("kind") == "unverified_lineage"
                for warning in check.warnings
                if isinstance(warning, dict)
            )
        )

    def test_joins_without_relationship_keys_are_explicitly_unverified(self):
        context, schemas = lab()
        for condition in ("TRUE", "l.account_id > a.account_id", "a.account_id=1"):
            with self.subTest(condition=condition):
                check = preflight_sql(
                    f"SELECT a.name FROM ledger l JOIN accounts a ON {condition}",
                    schemas,
                    context,
                    sample_bindings(),
                )
                self.assertFalse(check.all_join_keys_resolved)
                self.assertTrue(
                    any(
                        warning.get("kind") == "unverified_join"
                        for warning in check.warnings
                        if isinstance(warning, dict)
                    )
                )

    def test_scalar_snapshot_filter_does_not_create_an_unverified_join(self):
        context, schemas = lab()
        check = preflight_sql(
            "SELECT a.name FROM ledger l JOIN accounts a ON l.account_id=a.account_id "
            "WHERE l.period=(SELECT MAX(period) FROM ledger)",
            schemas,
            context,
            sample_bindings(),
        )
        self.assertTrue(check.all_join_keys_resolved)

    def test_workspace_business_rule_prevents_holdings_to_scheme_fanout(self):
        scope = resolve_scope("2", "2")
        context = create_context(source_ids=["smartwealth"])
        schemas = {
            name: context.table(name).schema()
            for name in scope.workspace.sources["smartwealth"].tables
        }
        bindings = configured_joins(
            scope.workspace.sources,
            get_domain_profile("2", tenant_id="2").join_semantics,
        )
        with self.assertRaises(SQLPreflightError) as raised:
            preflight_sql(
                "SELECT h.market_value FROM sw_holdings h JOIN sw_schemes s ON h.amc_id=s.amc_id",
                schemas,
                context,
                bindings,
            )
        diagnostic = raised.exception.diagnostic
        self.assertEqual(diagnostic.phase, "join_keys")
        self.assertIn("scheme_id", diagnostic.issues[0].repair)
        self.assertIn("multiplies market values", diagnostic.issues[0].repair)
        self.assertTrue(
            any(
                binding.relationship_id == "holding_scheme"
                for binding in diagnostic.join_context
            )
        )

    def test_wrong_entity_ids_receive_the_configured_bridge_join_path(self):
        scope = resolve_scope("1", "1")
        context = create_context(source_ids=["credit_cards"])
        schemas = {
            name: context.table(name).schema()
            for name in scope.workspace.sources["credit_cards"].tables
        }
        bindings = configured_joins(
            scope.workspace.sources,
            get_domain_profile("1", tenant_id="1").join_semantics,
        )
        with self.assertRaises(SQLPreflightError) as raised:
            preflight_sql(
                "SELECT c.customer_id FROM cc_customers c JOIN cc_monthly_financials f ON c.customer_id=f.card_id",
                schemas,
                context,
                bindings,
            )
        repair = raised.exception.diagnostic.issues[0].repair
        self.assertIn("cc_credit_cards.customer_id = cc_customers.customer_id", repair)
        self.assertIn("cc_monthly_financials.card_id = cc_credit_cards.card_id", repair)


BROKEN = """WITH latest AS (
 SELECT card_id,MAX(month) AS month FROM cc_monthly_financials GROUP BY card_id
) SELECT f.approved_net_spend_inr,f.profit_inr,f.opex_inr FROM latest f"""
FIXED = "SELECT SUM(approved_net_spend_inr) AS spend,SUM(profit_inr) AS profit,SUM(opex_inr) AS opex FROM cc_monthly_financials"


class SQLRetryTests(unittest.IsolatedAsyncioTestCase):
    def test_datafusion_type_function_errors_supply_valid_cast_syntax(self):
        context = SessionContext()
        for name in ("DATE", "TIMESTAMP"):
            with (
                self.subTest(name=name),
                self.assertRaises(SQLPreflightError) as raised,
            ):
                preflight_sql(f"SELECT {name}('2026-01-01') AS day", {}, context)
            diagnostic = raised.exception.diagnostic
            self.assertEqual(diagnostic.phase, "datafusion_compile")
            issue = next(
                issue
                for issue in diagnostic.issues
                if issue.kind == "unsupported_type_function"
            )
            self.assertIn("CAST(", issue.repair)
            self.assertIn(
                "DATE(...) and TIMESTAMP(...) are both unsupported", issue.repair
            )

    def setUp(self):
        patcher = patch.dict(os.environ, {"AGENT_TOOL_RETRIES": "3"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def sources(self):
        return DataSources(create_context(source_ids=["credit_cards"]), "1", "1")

    async def test_next_http_request_receives_scope_repairs_and_duplicates_do_not_execute(
        self,
    ):
        sources = self.sources()
        requests = []
        output = io.StringIO()
        actions = [
            ("describe_sources", {"step_id": "s1"}),
            ("run_sql", {"step_id": "s2", "statement": BROKEN}),
            ("run_sql", {"step_id": "s2", "statement": "-- repeated\n" + BROKEN}),
            ("run_sql", {"step_id": "s2", "statement": FIXED}),
        ]

        def respond(request):
            body = json.loads(request.content)
            index = len(requests)
            requests.append(body)
            if index in (2, 3):
                self.assertEqual(body["messages"][-1]["role"], "user")
                self.assertIn(
                    "Host validation repair handoff", body["messages"][-1]["content"]
                )
                self.assertIn("sql_generator", body["messages"][-1]["content"])
                self.assertIn(
                    "f.approved_net_spend_inr", body["messages"][-1]["content"]
                )
                last_tool = next(
                    message
                    for message in reversed(body["messages"])
                    if message["role"] == "tool"
                )
                envelope = json.loads(last_tool["content"])
                feedback = json.loads(envelope["error"])
                self.assertEqual(feedback["phase"], "qualification")
                self.assertEqual(
                    {issue["reference"] for issue in feedback["issues"]},
                    {"f.approved_net_spend_inr", "f.profit_inr", "f.opex_inr"},
                )
                self.assertEqual(feedback["repeated_failed_statement"], index == 3)
                self.assertIn("sql_correction", str(body["messages"]))
            if index == 1:
                schema = json.loads(body["messages"][-1]["content"])
                binding = next(
                    item
                    for item in schema["_sql_guidance"]["relationships"]
                    if item["relationship_id"] == "financial_profile"
                )
                self.assertEqual(binding["column_pairs"], [["card_id", "card_id"]])
                self.assertEqual(
                    binding["business"]["left_grain"], ["month", "card_id"]
                )
            if index == 4:
                self.assertNotIn('"sql_correction"', str(body["messages"][0]))
            return (
                completion(body, call=actions[index])
                if index < len(actions)
                else completion(body, content="Completed")
            )

        token = TRACE_ENABLED.set(True)
        try:
            async with mock_provider(respond):
                with (
                    redirect_stderr(output),
                    patch("execution.preflight_sql", wraps=preflight_sql) as preflight,
                    patch.object(sources, "query", wraps=sources.query) as queries,
                ):
                    result = await execute_plan(
                        "Aggregate source metrics",
                        ExecutionPlan.model_validate(executable_plan()),
                        sources,
                        configured_model("1", "1"),
                    )
        finally:
            TRACE_ENABLED.reset(token)
        self.assertEqual(preflight.call_count, 2)
        self.assertEqual(queries.call_count, 1)
        self.assertEqual(result.queries, [FIXED])
        self.assertIn("SQL_REJECTED:", output.getvalue())
        self.assertIn("unchanged rejected SQL", output.getvalue())
        self.assertEqual(result.completed_steps, ["s1", "s2"])

    async def test_transient_execution_failure_can_retry_identical_sql(self):
        sources = self.sources()
        original_query = sources.query
        calls = []

        def model(messages, info):
            index = len(calls)
            calls.append(1)
            if index == 0:
                name, args = "describe_sources", {"step_id": "s1"}
            elif index in (1, 2):
                if index == 2:
                    feedback = json.loads(tool_returns(messages)[-1].content)
                    self.assertEqual(feedback["phase"], "execution")
                    self.assertFalse(feedback["cacheable"])
                name, args = "run_sql", {"step_id": "s2", "statement": FIXED}
            else:
                return ModelResponse(parts=[TextPart("Completed")])
            return ModelResponse(parts=[ToolCallPart(name, args)])

        count = 0

        def transient(statement):
            nonlocal count
            count += 1
            if count == 1:
                raise RuntimeError("Temporary storage failure")
            return original_query(statement)

        with patch.object(sources, "query", side_effect=transient):
            result = await execute_plan(
                "Aggregate source metrics",
                ExecutionPlan.model_validate(executable_plan()),
                sources,
                FunctionModel(model),
            )
        self.assertEqual(count, 2)
        self.assertEqual(len(result.queries), 1)

    async def test_failure_cache_does_not_leak_between_requests(self):
        def make_model():
            calls = []

            def model(messages, info):
                index = len(calls)
                calls.append(1)
                actions = [
                    ("describe_sources", {"step_id": "s1"}),
                    ("run_sql", {"step_id": "s2", "statement": BROKEN}),
                    ("run_sql", {"step_id": "s2", "statement": FIXED}),
                ]
                if index < len(actions):
                    name, args = actions[index]
                    return ModelResponse(parts=[ToolCallPart(name, args)])
                return ModelResponse(parts=[TextPart("Completed")])

            return FunctionModel(model)

        with patch("execution.preflight_sql", wraps=preflight_sql) as preflight:
            for _ in range(2):
                await execute_plan(
                    "Aggregate source metrics",
                    ExecutionPlan.model_validate(executable_plan()),
                    self.sources(),
                    make_model(),
                )
        self.assertEqual(preflight.call_count, 4)
