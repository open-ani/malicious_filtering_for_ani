from fastapi import FastAPI
from .api.routes import router

SECRET_KEY = ""

app = FastAPI()

app.include_router(router)
