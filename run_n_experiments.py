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
         base_model: str = typer.Option(None, '--base_model', '-bm', help='pretrained model'),
         do_lower_case: bool = typer.Option(False, '--do_lower_case',  help='use lowercase in tokenizer'),
         batch_size: int = typer.Option(None, '--batch_size', '-bs', help='change batch_size used in base config'),
         pool_mem: bool = typer.Option(False, '--pool_mem', help='get pooled output as [max(mem),mean(mem)]'),
         only_head: bool = typer.Option(False, '--only_head', help='train only head'),
         random_init: bool = typer.Option(False, '--random_init', help='train from full random initialization'),
         warmup_steps: int = typer.Option(0, '--warmup_steps', help='set warm-up steps'),
         mean_max_pool: bool = typer.Option(False, '--mean_max_pool', help='use mean & max_pool as pooler in BERT'),
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

    if pool_mem:
        if mem_size == 0:
            print(f'pool_mem is set to {pool_mem}, but mem_size is {mem_size}')
            exit(1)
        config, _ = change_component(config, 'torch_bert_classifier', 'mem_size', mem_size)
        config, _ = change_component(config, 'torch_bert_classifier', 'pool_mem_tokens', pool_mem)
        print(f'pool_mem_tokens is set to {pool_mem}')

    if only_head:
        config, _ = change_component(config, 'torch_bert_classifier', 'only_head', only_head)

    if random_init:
        config, _ = change_component(config, 'torch_bert_classifier', 'random_init', random_init)

    if warmup_steps > 0:
        config, _ = change_component(config, 'torch_bert_classifier', 'warmup_steps', warmup_steps)
        config['train']['warmup_steps'] = warmup_steps

    if mean_max_pool:
        config, _ = change_component(config, 'torch_bert_classifier', 'mean_max_pool', mean_max_pool)

    # get model path
    if mem_size_changed:
        config['metadata']['variables']['MODEL_PATH'] += f'_{mem_size}'
    if pool_mem:
        config['metadata']['variables']['MODEL_PATH'] += f'_pool_mem'
    if base_model:
        config['metadata']['variables']['MODEL_PATH'] += f'_{base_model}'
    if only_head:
        config['metadata']['variables']['MODEL_PATH'] += f'_only_head'
    if random_init:
        config['metadata']['variables']['MODEL_PATH'] += f'_rnd'
    if mean_max_pool:
        config['metadata']['variables']['MODEL_PATH'] += f'_pool_meanmax'

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
    json.dump(total_metrics, (model_path / 'metrics.json').open('w'), indent=2)
    for mode in total_metrics:
        print(mode)
        for metric in total_metrics[mode]:
            mean = np.mean(total_metrics[mode][metric])
            std = np.std(total_metrics[mode][metric])
            m = np.max(total_metrics[mode][metric])
            print(f'\t{metric} {mean:.4f} +- {std:.4f}, max: {m:.4f}')


if __name__ == '__main__':
    typer.run(main)
