import abc
from abc import ABCMeta
from typing import List

from routing_schemas import TextsInput, PredictionsOutput


class BaseDanmakuFilterer(ABCMeta):

    @abc.abstractmethod
    def predict(cls, texts: TextsInput) -> PredictionsOutput:
        pass


