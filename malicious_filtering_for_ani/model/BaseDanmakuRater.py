import abc
import logging
from abc import ABCMeta
import asyncio

import torch

from transformers.models.bert import BertTokenizer, BertForSequenceClassification
from .routing_schemas import TextsInput, PredictionsOutput

logging.basicConfig(filename="model.log", level=logging.INFO)
logger = logging.getLogger(__name__)

model_semaphore = asyncio.Semaphore(1)

class BaseDanmakuFilterer(metaclass=ABCMeta):

    @abc.abstractmethod
    def predict(self, texts: TextsInput) -> PredictionsOutput:
        pass


def load_pretrained_tokenizer_model_transformer():
    tokenizer = BertTokenizer.from_pretrained('thu-coai/roberta-base-cold')
    model = BertForSequenceClassification.from_pretrained('thu-coai/roberta-base-cold')

    return tokenizer, model


class COLDDatabseDanmakuFilterer(BaseDanmakuFilterer):

    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = load_pretrained_tokenizer_model_transformer()
        self.model.to(self.device)

    async def predict(self, texts: TextsInput) -> PredictionsOutput:
        async with model_semaphore:
            model_input = self.tokenizer(texts.texts,return_tensors="pt",padding=True)
            model_input = {k: v.to(self.device) for k, v in model_input.items()}
            model_output = self.model(**model_input, return_dict=False)
            logger.info(model_output)

            prediction = torch.argmax(model_output[0].cpu(), dim=-1)
            res_dict = {"predictions": list(zip(texts.texts, prediction))}
            res = PredictionsOutput(**res_dict)
            return res


