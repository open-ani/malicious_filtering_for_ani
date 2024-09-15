from pydantic import BaseModel, model_validator
from typing import List

# This schema is to be changed to match the input and output of the model and the client.

class TextsInput(BaseModel):
    texts: List[str]

    @model_validator(mode='after')
    def check_list_not_empty(self):
        if self.texts is None or len(self.texts) == 0:
            raise ValueError('texts list cannot be empty')
        return self

class PredictionsOutput(BaseModel):
    predictions: List[(str, float)]

    @model_validator(mode='after')
    def check_list_not_empty(self):
        if self.predictions is None or len(self.predictions) == 0:
            raise ValueError('predictions list cannot be empty')
        return self
