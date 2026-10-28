import asyncio
import os
import unittest
from unittest.mock import patch
from pipeline import Pipeline, PipelineStage, PipelineRequest, PipelineServices
from runtime import model_settings, MAX_TOKENS_OVERRIDE

class TokenBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_overrides_are_isolated_and_reset_on_failure(self):
        class BudgetStage(PipelineStage):
            name = 'budget'
            async def process(self, context):
                await asyncio.sleep(0)
                context.answer = str(model_settings()['max_tokens'])
                if context.request.question == 'fail': raise RuntimeError('example')
        services = PipelineServices()
        pipeline = Pipeline(services, stages=[BudgetStage(services)])
        with patch('runtime.load_dotenv'), patch.dict(os.environ, {'LITELLM_MAX_OUTPUT_TOKENS':'5000'}):
            results = await asyncio.gather(*[pipeline.run(PipelineRequest('check','1','1',max_tokens=n))
                                            for n in (8192,16384,None)])
            self.assertEqual([r.answer for r in results], ['8192','16384','5000'])
            with self.assertRaises(RuntimeError):
                await pipeline.run(PipelineRequest('fail','1','1',max_tokens=8192))
            self.assertIsNone(MAX_TOKENS_OVERRIDE.get())
            self.assertEqual(model_settings()['max_tokens'],5000)
            for invalid in (0,-1,True,'8192'):
                with self.assertRaises(ValueError):
                    await pipeline.run(PipelineRequest('check','1','1',max_tokens=invalid))
