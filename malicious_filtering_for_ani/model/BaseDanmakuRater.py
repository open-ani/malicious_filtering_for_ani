import abc
import logging
from abc import ABCMeta

import torch

from transformers.models.bert import BertTokenizer, BertForSequenceClassification
from routing_schemas import TextsInput, PredictionsOutput

logging.basicConfig(filename="model.log", level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.tokenizer, self.model = load_pretrained_tokenizer_model_transformer()

    def predict(self, texts: TextsInput) -> PredictionsOutput:
        model_input = self.tokenizer(texts.texts,return_tensors="pt",padding=True)
        model_output = self.model(**model_input, return_dict=False)
        logger.info(model_output)

        prediction = torch.argmax(model_output[0].cpu(), dim=-1)
        res_dict = {"predictions": list(zip(texts.texts, prediction))}
        res = PredictionsOutput(**res_dict)
        return res


