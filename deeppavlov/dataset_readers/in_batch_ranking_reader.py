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

import copy
import json
import pickle

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader


@register('in_batch_ranking_reader')
class InBatchRankingReader(DatasetReader):
    """Class to read training datasets in OntoNotes format"""

    def read(self, data_path: str, train_size: int = None, valid_size: int = None):
        if str(data_path).endswith(".pickle"):
            with open(data_path, 'rb') as fl:
                dataset = pickle.load(fl)
        elif str(data_path).endswith(".json"):
            with open(data_path, 'r') as fl:
                dataset = json.load(fl) 
        
        if "valid" not in dataset and "test" in dataset:
            dataset["valid"] = copy.deepcopy(dataset["test"])
        if train_size is not None:
            dataset["train"] = dataset["train"][:train_size]
        if valid_size is not None:
            dataset["valid"] = dataset["valid"][:valid_size]
            dataset["test"] = dataset["test"][:valid_size]

        return dataset

