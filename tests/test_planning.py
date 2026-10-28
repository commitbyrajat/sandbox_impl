"""Verify classification/planning separation and execution of ordered steps."""

from contextlib import redirect_stderr
import io
import json
import unittest
from unittest.mock import patch

from pydantic_ai.messages import TextPart, ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agent import classify_question, plan_question
from test_support import ask
from planning import ExecutionPlan, QuestionIntent
from test_agent import executable_plan, structured_response, tool_returns


class PlanningTests(unittest.IsolatedAsyncioTestCase):
    async def test_classifier_only_identifies_task(self):
        def classifier(messages, info):
            self.assertNotIn('cc_transactions', info.instructions)
            self.assertNotIn('sw_holdings', info.instructions)
            self.assertEqual(info.function_tools, [])
            properties = info.output_tools[0].parameters_json_schema['properties']
            self.assertEqual(set(properties), {'intent', 'objective', 'constraints', 'clarification', 'objectives'})
            return structured_response(info, {
                'intent': 'aggregation', 'objective': 'Total approved card spending',
                'constraints': ['INR only', 'Approved transactions only'],
            })
        with patch('data_access.create_context') as loader:
            intent = await classify_question('Total approved INR card spending', tenant_id='1', workspace_id='1',
                                             model=FunctionModel(classifier))
            loader.assert_not_called()
        self.assertEqual(intent.constraints, ['INR only', 'Approved transactions only'])

    async def test_classify_then_plan_before_opening_sources(self):
        events = []

        def classifier(messages, info):
            events.append('classify')
            return structured_response(info, {
                'intent': 'aggregation', 'objective': 'Count card transactions',
            })

        def planner(messages, info):
            events.append('plan')
            self.assertEqual(events, ['classify', 'plan'])
            self.assertIn('cc_transactions', info.instructions)
            self.assertNotIn('sw_holdings', info.instructions)
            prompt = json.loads(messages[-1].parts[0].content)
            self.assertEqual(prompt['classified_intent']['objective'], 'Count card transactions')
            return structured_response(info, executable_plan())

        with patch('data_access.create_context') as loader:
            intent, plan = await plan_question(
                'How many transactions?', tenant_id='1', workspace_id='1', model=FunctionModel(planner),
                classifier_model=FunctionModel(classifier),
            )
            loader.assert_not_called()
        self.assertEqual(events, ['classify', 'plan'])
        self.assertEqual([s.action for s in plan.steps], ['schema', 'sql'])

    async def test_step_order_failed_query_retry_and_completion(self):
        calls = []

        def classifier(messages, info):
            return structured_response(info, {'intent': 'aggregation', 'objective': 'Count customers'})

        def planner(messages, info):
            return structured_response(info, executable_plan())

        def executor(messages, info):
            index = len(calls)
            calls.append(index)
            if index == 0:
                # Final output is rejected before the plan has run.
                return ModelResponse(parts=[TextPart('There are 20 customers.')])
            if index == 1:
                # SQL is rejected until schema inspection completes.
                return ModelResponse(parts=[ToolCallPart('run_sql', {
                    'step_id': 's2', 'statement': 'SELECT count(*) AS n FROM cc_customers',
                })])
            if index == 2:
                return ModelResponse(parts=[ToolCallPart('describe_sources', {'step_id': 's1'})])
            if index == 3:
                # Failure leaves s2 pending, allowing a corrected query.
                return ModelResponse(parts=[ToolCallPart('run_sql', {
                    'step_id': 's2', 'statement': 'SELECT missing_column FROM cc_customers',
                })])
            if index == 4:
                return ModelResponse(parts=[ToolCallPart('run_sql', {
                    'step_id': 's2', 'statement': 'SELECT count(*) AS n FROM cc_customers',
                })])
            evidence = json.loads(tool_returns(messages)[-1].content)
            self.assertEqual(evidence['rows'], [{'n': 20}])
            return ModelResponse(parts=[TextPart('20 credit card customers.')])

        trace = io.StringIO()
        with redirect_stderr(trace):
            answer = await ask('How many customers?', tenant_id='1', workspace_id='1',
                               model=FunctionModel(executor), classifier_model=FunctionModel(classifier),
                               planner_model=FunctionModel(planner), trace=True)
        self.assertEqual(answer, '20 credit card customers.')
        self.assertEqual(len(calls), 6)
        self.assertIn('COMPLETED: ["s1", "s2"]', trace.getvalue())
        self.assertIn('INTENT:', trace.getvalue())
        self.assertEqual(trace.getvalue().count('SQL: '), 1)

    async def test_plan_rejects_invalid_order_and_unplanned_python(self):
        plan = executable_plan('1', 'sql_python')
        plan['steps'][1], plan['steps'][2] = plan['steps'][2], plan['steps'][1]
        with self.assertRaises(ValueError):
            ExecutionPlan.model_validate(plan)
        plan = executable_plan('1', 'sql_python')
        plan['execution_mode'] = 'sql'
        with self.assertRaises(ValueError):
            ExecutionPlan.model_validate(plan)
        plan = executable_plan()
        plan['steps'][1]['step_id'] = 's1'
        with self.assertRaises(ValueError):
            ExecutionPlan.model_validate(plan)

    async def test_classifier_rejects_routing_fields(self):
        with self.assertRaises(ValueError):
            QuestionIntent.model_validate({
                'intent': 'aggregation', 'objective': 'Count customers',
                'sources': ['smartwealth'], 'execution_mode': 'sql',
            })
