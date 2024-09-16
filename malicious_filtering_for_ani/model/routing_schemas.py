from pydantic import BaseModel, model_validator
from typing import List, Tuple

class TextsInput(BaseModel):
    texts: List[str]

    @model_validator(mode='after')
    def check_list_not_empty(self):
        if self.texts is None or len(self.texts) == 0:
            raise ValueError('texts list cannot be empty')
        return self

class PredictionsOutput(BaseModel):
    predictions: List[Tuple[str, float]]

    @model_validator(mode='after')
    def check_list_not_empty(self):
        if self.predictions is None or len(self.predictions) == 0:
            raise ValueError('predictions list cannot be empty')
        return self
