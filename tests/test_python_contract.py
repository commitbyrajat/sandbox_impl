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
