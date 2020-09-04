import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm as tqdm

from deeppavlov import evaluate_model, train_model


def change_component(config, component, field, value):
    value_set = False
    for i in range(len(config['chainer']['pipe'])):
        if config['chainer']['pipe'][i]['class_name'] == component:
            config['chainer']['pipe'][i][field] = value
            value_set = True
            break
    return config, value_set


def main(config: str = typer.Argument(..., help='config to run experiment'),
         n_runs: int = typer.Option(5, '--n_runs', '-n', help='n runs of experiment'),
         mem_size: int = typer.Option(0, '--mem_size', '-m', help='change mem_size in mem config'),
         base_model: str = typer.Option(None, '--base_model', '-m', help='pretrained model'),
         do_lower_case: bool = typer.Option(False, '--do_lower_case',  help='use lowercase in tokenizer'),
         batch_size: int = typer.Option(None, '--batch_size', '-bs', help='change batch_size used in base config')
         ) -> None:
    print(f'Running {n_runs} experiments for {config}')
    config = json.load(open(config, 'r'))
    # set mem_size
    mem_size_changed = False
    if mem_size != 0:
        config, mem_size_changed = change_component(config, 'torch_mem_tokens_preprocessor', 'mem_size', mem_size)
        if not mem_size_changed:
            print('config file does not support mem_size argument')
            exit(1)
        else:
            print(f'mem_size is set to {mem_size}')

    if base_model:
        config['metadata']['variables']['BASE_MODEL'] = base_model
        config, _ = change_component(config, 'torch_bert_preprocessor', 'do_lower_case', do_lower_case)
        print(f'base model is set to {base_model}')

    if batch_size:
        config['train']['batch_size'] = batch_size

    # get model path
    if mem_size_changed:
        config['metadata']['variables']['MODEL_PATH'] += f'_{mem_size}'
    if base_model:
        config['metadata']['variables']['MODEL_PATH'] += f'_{base_model}'
    model_path = config['metadata']['variables']['MODEL_PATH']
    while '{' in model_path and '}' in model_path:
        model_path = model_path.format(**config['metadata']['variables'])
    model_path = Path(model_path).expanduser()
    print(config)
    # run experiments
    total_metrics = defaultdict(dict)
    for _ in tqdm(range(n_runs)):
        # remove previously trained model
        if model_path.exists():
            shutil.rmtree(model_path)
        # write current config file
        model_path.mkdir(parents=True)
        json.dump(config, (model_path / 'config.json').open('w'), indent=2)
        _ = train_model(config)
        metrics = evaluate_model(config)
        for mode in metrics:
            for m in metrics[mode]:
                if m in total_metrics[mode]:
                    total_metrics[mode][m] += [metrics[mode][m]]
                else:
                    total_metrics[mode][m] = [metrics[mode][m]]
    # print stat
    print(config)
    print('-' * 15)
    print(total_metrics)
    for mode in total_metrics:
        print(mode)
        for metric in total_metrics[mode]:
            mean = np.mean(total_metrics[mode][metric])
            std = np.std(total_metrics[mode][metric])
            m = np.max(total_metrics[mode][metric])
            print(f'\t{metric} {mean:.4f} +- {std:.4f}, max: {m:.4f}')


if __name__ == '__main__':
    typer.run(main)
