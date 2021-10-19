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

import random
from logging import getLogger
from typing import List

from sklearn.model_selection import train_test_split

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

log = getLogger(__name__)


@register('copy_domain_iterator')
class CopyDomainIterator(DataLearningIterator):

    def __init__(self, data: dict,
                 fields_to_merge: List[str] = None, merged_field: str = None,
                 field_to_split: str = None, split_fields: List[str] = None, split_proportions: List[float] = None,
                 seed: int = None, shuffle: bool = True, split_seed: int = None,
                 stratify: bool = None,
                 *args, **kwargs):
        """
        Initialize dataset using data from DatasetReader,
        merges and splits fields according to the given parameters.
        """
        super().__init__(data, seed=seed, shuffle=shuffle)
        
        batch_size = 10
        self.entity_samples = [elem for elem in self.data["train"] if elem[1][0] == 1]
        self.batches = {}
        train_batches = []
        num_batches = len(self.data["train"]) // batch_size + int(len(self.data["train"]) % batch_size > 0)
        for i in range(num_batches):
            train_batches.append(self.data["train"][i*batch_size:(i+1)*batch_size])
        entity_samples = [elem for elem in self.data["train"] if elem[1][0] == 1]
        print("entity samples", len(entity_samples))
        num_batches = len(entity_samples) // batch_size + int(len(entity_samples) % batch_size > 0)
        for i in range(num_batches):
            train_batches.append(entity_samples[i*batch_size:(i+1)*batch_size])
        random.shuffle(train_batches)
        self.batches["train"] = train_batches
        
        valid_batches = []
        num_batches = len(self.data["valid"]) // batch_size + int(len(self.data["valid"]) % batch_size > 0)
        for i in range(num_batches):
            valid_batches.append(self.data["valid"][i*batch_size:(i+1)*batch_size])
        self.batches["valid"] = valid_batches

    def gen_batches(self, batch_size: int, data_type: str = 'train',
                    shuffle: bool = None):
        
        for batch in self.batches[data_type]:
            yield tuple(zip(*batch))
