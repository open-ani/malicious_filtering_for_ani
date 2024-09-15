from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import asyncio
import msgpack
import logging

from ...models.routing_schemas import TextsInput

app = FastAPI()

logging.basicConfig(filename="api.log", level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/v1/filter")
async def predict(file: UploadFile):
    try:
        binary_content = await file.read()

        texts = msgpack.unpackb(binary_content)
        input_data = TextsInput(**texts)
        logger.info(f"Received {len(texts)} texts for prediction.")

        predictions = await model.predict(texts)
        binary_predictions = msgpack.packb(predictions.model_dump(), use_bin_type=True)

        return Response(content=binary_predictions, media_type='application/octet-stream')

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
