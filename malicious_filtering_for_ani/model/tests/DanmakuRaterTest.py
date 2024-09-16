import unittest
import asyncio

from ..BaseDanmakuRater import COLDDatabseDanmakuFilterer
from ..routing_schemas import TextsInput, PredictionsOutput

class DanmakuRaterTest(unittest.TestCase):

    def test_rate(self):
        asyncio.run(self.async_test_rate())

    async def async_test_rate(self):
        model = COLDDatabseDanmakuFilterer()
        text = {"texts": ["我是好人", "你是傻逼"]}
        input_data = TextsInput(**text)
        result = await model.predict(input_data)
        self.assertIsInstance(result, PredictionsOutput)
        print(result.model_dump())
