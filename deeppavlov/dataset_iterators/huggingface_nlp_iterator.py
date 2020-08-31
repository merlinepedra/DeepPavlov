# Copyright 2020 Neural Networks and Deep Learning lab, MIPT
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

from typing import List, Tuple, Any, Union

from nlp import Dataset

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('huggingface_nlp_iterator')
class HuggingFaceNLPIterator(DataLearningIterator):

    def preprocess(self, data: Dataset,
                   features: Union[str, List[str]], label: str = 'label', use_label_name: bool = True,
                   *args, **kwargs) -> List[Tuple[Any, Any]]:
        dataset = []
        for example in data:
            if isinstance(features, str):
                feat = example[features]
            else:
                feat = tuple(example[f] for f in features)
            lb = example[label]
            if use_label_name:
                # -1 label is used if there is no label
                lb = data.info.features[label].names[lb] if lb != -1 else lb
            dataset += [(feat, lb)]
        return dataset
