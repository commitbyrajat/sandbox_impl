import unittest
from pathlib import Path
import sys
from execution import prepare_python
from pydantic_monty import AsyncMonty

class PythonContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_print_retains_structured_result(self):
        code = prepare_python("rows = data['q']\nresult = {'total': sum(r['value'] for r in rows)}\nprint(result)")
        async with AsyncMonty(binary_path=str(Path(sys.executable).parent/'monty'), max_processes=1) as pool:
            async with pool.checkout() as session:
                result = await session.feed_run(code, inputs={'data': {'q': [{'value': 7}]}})
        self.assertEqual(result, {'total': 7})

    async def test_copied_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Read SQL rows'):
            prepare_python("rows = [('AMC', 100)]\nprint(rows)")

    async def test_missing_decimal_import_in_callback_is_repaired(self):
        from decimal import Decimal
        code = prepare_python("rows = data['q']\n"
            "groups = [{'value': Decimal('0')}, {'value': Decimal('10')}]\n"
            "target = min(groups, key=lambda g: Decimal(g['value']))\n"
            "target['value'] += rows[0]['value']\n"
            "groups")
        self.assertIn('from decimal import Decimal', code)
        async with AsyncMonty(binary_path=str(Path(sys.executable).parent/'monty'), max_processes=1) as pool:
            async with pool.checkout(limits={'max_suspensions':0}) as session:
                result = await session.feed_run(code, inputs={'data': {'q': [{'value': Decimal('7')}]}})
        self.assertEqual(result, [{'value':Decimal('7')}, {'value':Decimal('10')}])

    async def test_undefined_callback_names_fail_before_sandbox(self):
        with self.assertRaisesRegex(ValueError, 'Undefined sandbox names: missing_amount'):
            prepare_python("sorted(data['q'], key=lambda row: missing_amount(row))")

    async def test_defined_helpers_and_import_aliases_are_preserved(self):
        code = "from decimal import Decimal as D\ndef value(row):\n    return D(row['amount'])\nsorted(data['q'], key=value)"
        self.assertEqual(prepare_python(code), code)

    async def test_supplied_data_cannot_be_replaced_by_copied_rows(self):
        with self.assertRaisesRegex(ValueError, 'data is supplied by the host'):
            prepare_python("data = {'q': [{'value': 100}]}\ndata['q']")
        self.assertEqual(prepare_python("data = data['q']\ndata"), "data = data['q']\ndata")
