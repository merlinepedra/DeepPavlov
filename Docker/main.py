import argparse
import datetime
import os
import shutil
from logging import getLogger
from pathlib import Path

import pandas as pd
from filelock import FileLock

from deeppavlov import train_model, evaluate_model
from deeppavlov.core.commands.utils import parse_config, expand_path

DATA_PATH = Path('/data')
LOCKFILE = DATA_PATH / 'lockfile'
LOG_PATH = DATA_PATH / 'logs'
metrics_filename = DATA_PATH / "metrics_score_history.csv"
ner_config = parse_config("ner_rus_distilbert_torch.json")
logger = getLogger(__file__)


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
            if after_training:
                model_path = ner_config["metadata"]["variables"]["MODEL_PATH"]
                model_path_exp = str(expand_path(model_path))
                files = os.listdir(model_path_exp)
                new_model_path_exp = model_path_exp.strip("_new")
                for fl in files:
                    shutil.copy(f"{model_path_exp}/{fl}", new_model_path_exp)
                shutil.rmtree(model_path_exp)
    else:
        df = pd.DataFrame.from_dict({"time": [datetime.datetime.now()],
                                     "old_metric": [cur_f1],
                                     "new_metric": [cur_f1],
                                     "update_model": [after_training]})
    df.to_csv(metrics_filename, index=False)

    return cur_f1


def train(data_path: str = ''):
    if data_path:
        ner_config["dataset_reader"] = {
            "class_name": "sq_reader",
            "data_path": data_path
        }
    init_path = next(
        i for i in ner_config['metadata']['download'] if 'ner_rus_distilbert_torch.tar.gz' in i['url']
    )['subdir']
    model_path = ner_config["metadata"]["variables"]["MODEL_PATH"]
    new_model_path = Path(f'{model_path}_new')
    model_path = Path(model_path)

    if new_model_path.exists():
        shutil.rmtree(new_model_path)
    shutil.copytree(init_path, new_model_path)

    ner_config["metadata"]["variables"]["MODEL_PATH"] = str(new_model_path)

    for i in range(len(ner_config["chainer"]["pipe"])):
        if ner_config["chainer"]["pipe"][i].get("class_name", "") == "torch_transformers_sequence_tagger":
            ner_config['chainer']['pipe'][i]['load_path'] = ner_config['chainer']['pipe'][i]['save_path'] = \
                str(new_model_path)
            logger.warning(f"load and save path {new_model_path}")
    train_model(ner_config)
    logger.info('Training finished. Starting evaluation...')
    _ = evaluate(ner_config, True)
    logger.info('Evaluation finished.')
    shutil.rmtree(model_path)
    logger.info(f'{model_path} removed')
    new_model_path.rename(model_path)
    logger.info(f'{new_model_path} with trained model renamed to {model_path}')
    logger.info('Training finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, default='')
    args = parser.parse_args()
    with FileLock(str(LOCKFILE)):
        train(args.data)
    LOCKFILE.unlink()
