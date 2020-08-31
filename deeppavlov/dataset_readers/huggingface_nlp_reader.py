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


from overrides import overrides
from typing import Dict, Optional

from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.common.registry import register

from nlp import load_dataset, Dataset


@register('huggingface_nlp_reader')
class HuggingFaceNLPDatasetReader(DatasetReader):
    """Adds HuggingFace NLP Datasets https://huggingface.co/nlp/ to DeepPavlov
    """

    @overrides
    def read(self, data_path: str, path: str, name: Optional[str] = None, train: str = 'train',
             valid: str = 'validation', test: str = 'test', **kwargs) -> Dict[str, Dataset]:
        """Wraps nlp.load_dataset method

        Args:
            data_path: DeepPavlov data_path argument, is not used, but passed by trainer
            path: nlp.load_dataset path argument
            name: nlp.load_dataset name argument
            train: split name to use as training data. Defaults to 'train'.
            valid: split name to use as validation data. Defaults to 'validation'.
            test: split name to use as test data. Defaults to 'test'.

        Returns:
            Dict[str, List[Dict]]: Dictionary with train, valid, test datasets
        """
        if 'split' in kwargs:
            raise RuntimeError('Split argument was used. Use train, valid, test arguments instead of split.')
        split = ['train', 'valid', 'test']
        split_mapping = {'train': train, 'valid': valid, 'test': test}
        # filter sets that are not available in dataset
        split_mapped = [split_mapping[s] for s in split if split_mapping[s]]
        dataset = load_dataset(path=path, name=name, split=split_mapped, **kwargs)
        return dict(zip(split, dataset))
