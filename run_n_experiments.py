import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm as tqdm

from deeppavlov import evaluate_model, train_model


def main(config: str = typer.Argument(..., help='config to run experiment'),
         n_runs: int = typer.Option(5, '--n_runs', '-n', help='n runs of experiment'),
         mem_size: int = typer.Option(0, '--mem_size', '-m', help='change mem_size in mem config')
         ) -> None:
    print(f'Running {n_runs} experiments for {config}')
    config = json.load(open(config, 'r'))
    # set mem_size
    mem_size_changed = False
    if mem_size != 0:
        for i in range(len(config['chainer']['pipe'])):
            if config['chainer']['pipe'][i]['class_name'] == 'torch_mem_tokens_preprocessor':
                config['chainer']['pipe'][i]['mem_size'] = mem_size
                mem_size_changed = True
                break
        if not mem_size_changed:
            print('config file does not support mem_size argument')
        else:
            print(f'mem_size is set to {mem_size}')

    # get model path
    model_path = config['metadata']['variables']['MODEL_PATH']
    while '{' in model_path and '}' in model_path:
        model_path = model_path.format(**config['metadata']['variables'])
    if mem_size_changed:
        model_path = f'{model_path}_{mem_size}'
    model_path = Path(model_path).expanduser()
    # run experiments
    total_metrics = defaultdict(dict)
    for _ in tqdm(range(n_runs)):
        # remove previously trained model
        if model_path.exists():
            shutil.rmtree(model_path)
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
