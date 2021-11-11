import asyncio
import json
import os
import shutil
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
from deeppavlov.core.commands.utils import parse_config, expand_path
from deeppavlov import configs, build_model, train_model, evaluate_model

logger = getLogger(__file__)
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

ner_config = parse_config("ner_rus_distilbert_torch.json")
entity_detection_config = parse_config("ner_rus_vx_distil.json")
entity_detection = build_model(entity_detection_config, download=False)

@app.post("/model")
async def model(request: Request):
    while True:
        try:
            inp = await request.json()
            texts = inp["x"]
            entity_substr, entity_lemm_substr, entity_offsets, entity_init_offsets, tags, sentences_offsets, \
                sentences, probas = entity_detection(texts)
            res = {"entity_substr": entity_substr, "entity_lemm_substr": entity_lemm_substr,
                   "entity_offsets": entity_offsets, "entity_init_offsets": entity_init_offsets, "tags": tags,
                   "sentences_offsets": sentences_offsets, "sentences": sentences, "probas": probas}
            return res
            
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.warning(f'{host} is unavailable, restarting worker container')
            loop = asyncio.get_event_loop()
            loop.create_task(porter.update_container(host))

@app.post("/train")
async def model(request: Request):
    while True:
        try:
            inp = await request.json()
            train_filename = inp["train_filename"]
            with open(train_filename, 'r') as fl:
                total_data = json.load(fl)
            train_data = total_data[:int(len(total_data) * 0.9)]
            test_data = total_data[int(len(total_data) * 0.9):]
            logger.info(f"-------------- train data {len(train_data)} test data {len(test_data)}")
            logger.warning(f"-------------- train data {len(train_data)} test data {len(test_data)}")
            new_filename = f"{train_filename.strip('.json')}_train.json"
            with open(new_filename, 'w', encoding="utf8") as out:
                json.dump({"train": train_data, "valid": test_data, "test": test_data},
                          out, indent=2, ensure_ascii=False)
            
            ner_config["dataset_reader"] = {
                "class_name": "sq_reader",
                "data_path": new_filename
            }
            
            model_path = ner_config["metadata"]["variables"]["MODEL_PATH"]
            old_path = model_path.split("/")[-1]
            new_path = f"{old_path}_new"
            model_path_exp = str(expand_path(model_path))
            files = os.listdir(model_path_exp)
            if not os.path.exists(f'{model_path_exp}_new'):
                os.makedirs(f'{model_path_exp}_new')
            
            logger.info(f"-------------- model path exp {model_path_exp}")
            logger.warning(f"-------------- model path exp {model_path_exp}")
            
            for fl in files:
                shutil.copy(f"{model_path_exp}/{fl}", f'{model_path_exp}_new')
            
            ner_config["metadata"]["variables"]["MODEL_PATH"] = f"{model_path}_new"
            logger.info(f"-------------- model path {ner_config['metadata']['variables']['MODEL_PATH']}")
            logger.warning(f"-------------- model path {ner_config['metadata']['variables']['MODEL_PATH']}")
            for i in range(len(ner_config["chainer"]["pipe"])):
                if ner_config["chainer"]["pipe"][i].get("class_name", "") == "torch_transformers_sequence_tagger":
                    ner_config['chainer']['pipe'][i]['load_path'] = ner_config['chainer']['pipe'][i]['load_path'].replace(old_path, new_path)
                    ner_config['chainer']['pipe'][i]['save_path'] = ner_config['chainer']['pipe'][i]['save_path'].replace(old_path, new_path)
                    logger.info(f"-------------- load path {ner_config['chainer']['pipe'][i]['load_path']}")
                    logger.warning(f"-------------- load path {ner_config['chainer']['pipe'][i]['load_path']}")
                    logger.info(f"-------------- save path {ner_config['chainer']['pipe'][i]['save_path']}")
                    logger.warning(f"-------------- save path {ner_config['chainer']['pipe'][i]['save_path']}")
            
            #train_model(ner_config)
            res = evaluate_model(ner_config)
            metrics = dict(res["test"])
            
            return {"metrics": metrics}
            
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.warning(f'{host} is unavailable, restarting worker container')
            loop = asyncio.get_event_loop()
            loop.create_task(porter.update_container(host))

@app.post("/test")
async def model(request: Request):
    while True:
        try:
            inp = await request.json()
            test_filename = inp["test_filename"]
            with open(test_filename, 'r') as fl:
                test_data = json.load(fl)
            new_filename = f"{test_filename.strip('.json')}_test.json"
            with open(new_filename, 'w', encoding="utf8") as out:
                json.dump({"train": [], "valid": [], "test": test_data},
                          out, indent=2, ensure_ascii=False)
            
            ner_config["dataset_reader"] = {
                "class_name": "sq_reader",
                "data_path": new_filename
            }
            res = evaluate_model(ner_config)
            metrics = dict(res["test"])
            
            return {"metrics": metrics}
            
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.warning(f'{host} is unavailable, restarting worker container')
            loop = asyncio.get_event_loop()
            loop.create_task(porter.update_container(host))

uvicorn.run(app, host='0.0.0.0', port=8000)
