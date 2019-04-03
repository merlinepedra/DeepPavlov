# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import shutil
from collections import OrderedDict
from pathlib import Path
from shutil import rmtree
from typing import Union

from deeppavlov.pipeline_manager.plot_gen import get_met_info
from deeppavlov.pipeline_manager.plot_gen import plot_res
from deeppavlov.pipeline_manager.table_gen import build_pipeline_table
from deeppavlov.pipeline_manager.table_gen import sort_pipes


class ExperimentObserver:
    """
    Implements the functions of observing the course of experiments,
    collecting results, time and other useful information, logging and storing it.

    Args:
            name: name of the experiments.
            root: path to root folder.
            info: additional information that you want to add to the log, the content of the dictionary
             does not affect the algorithm
            date: date of the experiment.
            test_mode:
            plot:
    """

    def __init__(self,
                 name: str,
                 launch_name: Union[str, Path],
                 root: Union[str, Path],
                 info: dict,
                 date: str,
                 plot: bool,
                 test_mode: bool = False) -> None:
        """ Initializes the log, creates a folders tree and files necessary for the observer to work. """
        self.plot = plot
        self.test_mode = test_mode

        # build folder dependencies
        self.exp_log_path = root.joinpath(name, date)
        if test_mode:
            self.exp_log_path = self.exp_log_path.joinpath("tmp")
            if self.exp_log_path.exists():
                rmtree(str(self.exp_log_path))

        container = [x.name.split('_')[-1] for x in self.exp_log_path.glob(launch_name + "*")]
        if len(container) == 0:
            self.exp_log_path = self.exp_log_path.joinpath(launch_name)
        else:
            container.remove(launch_name)
            if len(container) == 0:
                self.exp_log_path = self.exp_log_path.joinpath(launch_name + '_2')
            else:
                self.exp_log_path = self.exp_log_path.joinpath(launch_name + f'_{max(int(x) for x in container) + 1}')

        self.exp_file = self.exp_log_path.joinpath(launch_name + '.json')
        self.log_file = self.exp_log_path.joinpath('logs.jsonl')

        self.save_path = self.exp_log_path.joinpath('checkpoints')
        self.save_path.mkdir(parents=True, exist_ok=True)
        if plot:
            self.exp_log_path.joinpath('images').mkdir(exist_ok=True)

        self.exp_info = OrderedDict(date=date,
                                    exp_name=launch_name,
                                    root=str(root),
                                    info=info,
                                    number_of_pipes=None,
                                    metrics=None,
                                    target_metric=None,
                                    dataset_name=None)
        self.log = None
        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None
        self.batch_size = None

    def save_exp_info(self, time: str) -> None:
        """
        Adding the time duration of the experiment in log file.

        Args:
            time: the time duration of the experiment

        Returns:
            None
        """
        self.exp_info['full_time'] = time
        with self.exp_file.open('w', encoding='utf8') as exp_file:
            json.dump(self.exp_info, exp_file)

    def tmp_reset(self) -> None:
        """ Reinitialize temporary attributes. """
        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None
        self.batch_size = None
        self.log = None

    def write(self) -> None:
        """ Write pipeline logs in jsonl file. """
        with self.log_file.open('a', encoding='utf8') as f:
            print(json.dumps(self.log), file=f)

    def update_log(self):
        """ Updates a log with new pipeline information. """
        for component in self.pipe_conf:
            if component.get('main') is True:
                self.model = component['component_name']

        pipe_name = '-->'.join([x['component_name'] for x in self.pipe_conf])

        self.log = {'pipe_index': self.pipe_ind,
                    'model': self.model,
                    'config': self.pipe_conf,
                    'light_config': pipe_name,
                    'time': self.pipe_time,
                    'batch_size': self.batch_size,
                    'results': self.pipe_res}

        self.write()
        self.tmp_reset()
        return self

    def save_config(self, conf: dict, ind: int) -> None:
        """ Save train config in checkpoint folder. """
        with self.save_path.joinpath(f'pipe_{ind}', 'config.json').open('w', encoding='utf8') as cf:
            json.dump(conf, cf)

    def save_best_pipe(self) -> None:
        """ Calculate the best pipeline and delete others pipelines checkpoints. """
        with self.exp_file.open('r', encoding='utf8') as info:
            exp_info = json.load(info)

        target_metric = exp_info['target_metric']
        dataset_name = exp_info['dataset_name']

        logs = []
        with self.log_file.open('r', encoding='utf8') as log_file:
            for line in log_file.readlines():
                logs.append(json.loads(line))
            sort_logs = sort_pipes(logs, target_metric)

        source = self.save_path / dataset_name
        dest1 = self.save_path / dataset_name / 'best_pipe'
        dest1.mkdir(exist_ok=True)

        files = source.iterdir()
        for f in files:
            if f.name.startswith('pipe') and f.name != 'pipe_{}'.format(sort_logs[0]['pipe_index']):
                rmtree((source / f.name))
            elif f.name == 'pipe_{}'.format(sort_logs[0]['pipe_index']):
                shutil.move(str(source / f.name), str(dest1))

    def build_report(self) -> None:
        """
        It builds a reporting table and a histogram of results for different models,
        based on data from the experiment log.

        Returns:
            None
        """
        logs = []
        with self.exp_file.open('r', encoding='utf8') as exp_log:
            exp_info = json.load(exp_log)

        with self.log_file.open('r', encoding='utf8') as log_file:
            for line in log_file.readlines():
                logs.append(json.loads(line))

        # create the xlsx file with results of experiments
        build_pipeline_table(exp_info, logs, save_path=self.exp_log_path)

        if self.plot:
            # scrub data from log for image creating
            info = get_met_info(logs)
            # plot histograms
            plot_res(info, exp_info['dataset_name'], self.exp_log_path.joinpath('images'))

    def build_pipe_checkpoint_folder(self, pipe, ind):
        save_path_i = self.save_path.joinpath("pipe_{}".format(ind + 1))
        save_path_i.mkdir()
        # save config in checkpoint folder
        self.save_config(pipe, ind + 1)
        return save_path_i

    def del_log(self):
        rmtree(str(self.exp_log_path))