import os
from fastapi import APIRouter, HTTPException, Response, Request, Depends, status
from fastapi.security.api_key import APIKeyHeader
import logging
import msgpack

from model.BaseDanmakuRater import COLDDatabseDanmakuFilterer
from model.routing_schemas import TextsInput

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise Exception("API_KEY environment variable not set")

API_KEY_NAME = "DanmakuFilter-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )


router = APIRouter()
logger = logging.getLogger(__name__)

model = COLDDatabseDanmakuFilterer()


@router.get("/test/")
async def test():
    return {"message": "Hello World"}


@router.post("/v1/filter")
async def predict(request: Request, api_key: str = Depends(get_api_key)):
    try:
        # Read and unpack the binary content
        binary_content = await request.body()
        texts = msgpack.unpackb(binary_content, raw=False)

        # Validate input data
        input_data = TextsInput(**texts)
        logger.info(f"Received {len(input_data.texts)} texts for prediction.")

        # Make predictions
        predictions = await model.predict(input_data)

        # Pack the response
        binary_predictions = msgpack.packb(predictions.model_dump(), use_bin_type=True)
        return Response(content=binary_predictions, media_type='application/octet-stream')
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
