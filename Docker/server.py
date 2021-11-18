import asyncio
import datetime
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List
from logging import getLogger

import aiohttp
import pandas as pd
import requests
import torch
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

metrics_filename = "/src/metrics_score_history.csv"
ner_config = parse_config("ner_rus_distilbert_torch.json")
entity_detection_config = parse_config("ner_rus_vx_distil.json")
entity_detection = build_model(entity_detection_config, download=True)

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


def evaluate(ner_config, after_training):
    res = evaluate_model(ner_config)
    logger.warning(f"metrics {res}")
    
    metrics = dict(res["test"])
    cur_f1 = metrics["ner_f1"]
    
    if Path(metrics_filename).exists():
        df = pd.read_csv(metrics_filename)
        max_metric = max(df["old_metric"].max(), df["new_metric"].max())
        if cur_f1 > max_metric:
            df = df.append({"time": datetime.datetime.now(),
                            "old_metric": max_metric,
                            "new_metric": cur_f1,
                            "update_model": after_training}, ignore_index=True)
    else:
        df = pd.DataFrame.from_dict({"time": [datetime.datetime.now()],
                                     "old_metric": [cur_f1],
                                     "new_metric": [cur_f1],
                                     "update_model": [after_training]})
    df.to_csv(metrics_filename, index=False)
    
    return cur_f1


@app.get('/last_train_metric')
async def get_metric(request: Request):
    while True:
        try:
            last_metrics = {}
            if Path(metrics_filename).exists():
                df = pd.read_csv(metrics_filename)
                last_metrics = df.iloc[-1].to_dict()
                logger.warning(f"last_metrics {last_metrics}")
            
            return {"success": True, "data": {"time": str(last_metrics["time"]),
                                              "old_metric": float(last_metrics["old_metric"]),
                                              "new_metric": float(last_metrics["new_metric"]),
                                              "update_model": bool(last_metrics["update_model"])}}
            
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.warning(f'{host} is unavailable, restarting worker container')
            loop = asyncio.get_event_loop()
            loop.create_task(porter.update_container(host))


def train(ner_config):
    train_model(ner_config)
    cur_f1 = evaluate(ner_config, True)
    torch.cuda.empty_cache()


@app.get("/train")
async def model_training(request: Request):
    while True:
        try:
            inp = await request.json()
            train_filename = inp["file"]
            with open(train_filename, 'r') as fl:
                total_data = json.load(fl)
            train_data = total_data[:int(len(total_data) * 0.9)]
            test_data = total_data[int(len(total_data) * 0.9):]
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
            
            logger.warning(f"-------------- model_path {model_path_exp} files {files}")
            
            #if os.path.isfile(f"{model_path_exp}_new"):
            #    os.remove(myfile)
            #if os.path.isdir(f"{model_path_exp}_new"):
            #    shutil.rmtree(f"{model_path_exp}_new")
            Path(f"{model_path_exp}_new").mkdir(parents=True, exist_ok=True)
            
            for fl in files:
                shutil.copy(f"{model_path_exp}/{fl}", f'{model_path_exp}_new')
            
            ner_config["metadata"]["variables"]["MODEL_PATH"] = f"{model_path}_new"
            logger.warning(f"-------------- model path {ner_config['metadata']['variables']['MODEL_PATH']}")
            for i in range(len(ner_config["chainer"]["pipe"])):
                if ner_config["chainer"]["pipe"][i].get("class_name", "") == "torch_transformers_sequence_tagger":
                    ner_config['chainer']['pipe'][i]['load_path'] = ner_config['chainer']['pipe'][i]['load_path'].replace(old_path, new_path)
                    ner_config['chainer']['pipe'][i]['save_path'] = ner_config['chainer']['pipe'][i]['save_path'].replace(old_path, new_path)
                    logger.warning(f"-------------- load path {ner_config['chainer']['pipe'][i]['load_path']}")
                    logger.warning(f"-------------- save path {ner_config['chainer']['pipe'][i]['save_path']}")
            
            threading.Thread(target=train, args=(ner_config,)).start()
            
            return {"success": True, "message": "Обучение инициировано"}
            
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.warning(f'{host} is unavailable, restarting worker container')
            loop = asyncio.get_event_loop()
            loop.create_task(porter.update_container(host))


@app.get("/evaluate")
async def model_testing(request: Request):
    while True:
        try:
            inp = await request.json()
            if inp is not None and isinstance(inp, dict) and inp.get("file", ""):
                test_filename = inp["file"]
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
            cur_ner_f1 = evaluate(ner_config, False)
            
            return {"metrics": cur_ner_f1}
            
        except aiohttp.client_exceptions.ClientConnectorError:
            logger.warning(f'{host} is unavailable, restarting worker container')
            loop = asyncio.get_event_loop()
            loop.create_task(porter.update_container(host))

uvicorn.run(app, host='0.0.0.0', port=8000)
