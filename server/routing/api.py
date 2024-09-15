from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import asyncio
import msgpack
import logging

app = FastAPI()

logging.basicConfig(filename="api.log", level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/v1/filter")
async def predict(file: UploadFile):
    try:
        binary_content = await file.read()

        texts = msgpack.unpackb(binary_content)
        if not isinstance(texts, list) or not all(isinstance(s, str) for s in texts):
            raise ValueError("Invalid input data format.")

        logger.info(f"Received {len(texts)} texts for prediction.")

        predictions = await model.predict(texts)
        binary_predictions = msgpack.packb(predictions, use_bin_type=True)

        return Response(content=binary_predictions, media_type='application/octet-stream')

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
