import asyncio
import json
from typing import Dict, List
from logging import getLogger

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse
from aliases import Aliases
from deeppavlov import configs, build_model

logger = getLogger(__file__)
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

ner = build_model(configs.ner.ner_rus_vx_distil, download=False)

@app.post("/model")
async def model(request: Request):
    while True:
        try:
            host = next(porter.active_hosts)
        except StopIteration:
            raise HTTPException(status_code=500, detail='No active workers')
        try:
            inp = await request.json()
            texts = inp["texts"]
            res = ner(texts)
            logger.warning(f"res {res}")
            logger.info(f"res {res}")
            
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.warning(f'{host} is unavailable, restarting worker container')
            loop = asyncio.get_event_loop()
            loop.create_task(porter.update_container(host))


uvicorn.run(app, host='0.0.0.0', port=8000)
