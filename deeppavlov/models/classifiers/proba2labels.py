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

from logging import getLogger
from typing import List, Union
from collections.abc import Iterable

import numpy as np

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.data.simple_vocab import SimpleVocabulary

log = getLogger(__name__)


@register('proba2labels')
class Proba2Labels(Component):
    """
    Class implements probability to labels processing using the following ways: \
     choosing one or top_n indices with maximal probability or choosing any number of indices \
      which probabilities to belong with are higher than given confident threshold

    Args:
        max_proba: whether to choose label with maximal probability
        confident_threshold: boundary probability value for sample to belong with the class (best use for multi-label)
        top_n: how many top labels with the highest probabilities to return

    Attributes:
        max_proba: whether to choose label with maximal probability
        confident_threshold: boundary probability value for sample to belong with the class (best use for multi-label)
        top_n: how many top labels with the highest probabilities to return
    """

    def __init__(self,
                 max_proba: bool = None,
                 confident_threshold: float = None,
                 top_n: int = None,
                 **kwargs) -> None:
        """ Initialize class with given parameters"""

        self.max_proba = max_proba
        self.confident_threshold = confident_threshold
        self.top_n = top_n

    def __call__(self, data: Union[np.ndarray, List[List[float]], List[List[int]]],
                 *args, **kwargs) -> Union[List[List[int]], List[int]]:
        """
        Process probabilities to labels

        Args:
            data: list of vectors with probability distribution

        Returns:
            list of labels (only label classification) or list of lists of labels (multi-label classification)
        """
        if self.confident_threshold:
            return [list(np.where(np.array(d) > self.confident_threshold)[0])
                    for d in data]
        elif self.max_proba:
            return [np.argmax(d, axis=-1) for d in data]
        elif self.top_n:
            return [np.argsort(d, axis=-1)[::-1][...,:self.top_n] for d in data]
        else:
            raise ConfigError("Proba2Labels requires one of three arguments: bool `max_proba` or "
                              "float `confident_threshold` for multi-label classification or"
                              "integer `top_n` for choosing several labels with the highest probabilities")


@register('proba2labelsdict')
class Proba2LabelsDict(Component):

    """
    Transforms the lists (arrays) of probabilities to dictionaries of the form 'label': 'prob'
    """

    def __init__(self,
                 vocab: SimpleVocabulary,
                 confidence_threshold: float = 0.0,
                 top_n: int = None,
                 return_mode = "dict",
                 **kwargs) -> None:
        self.vocab = vocab
        self.confidence_threshold = confidence_threshold
        self.top_n = top_n
        if return_mode in ["dict", "items"]:
            self.return_mode = return_mode
        else:
            raise ValueError("'return_mode' must be in list ['dict', 'items'].")

    def _process(self, data):
        if isinstance(data[0], (float, np.int_, np.float32)):
            labels = self.vocab._i2t
            if self.confidence_threshold > 0.0:
                indexes = np.where(data >= self.confidence_threshold)[0]
                labels = np.take(labels, indexes)
                data = np.take(data, indexes)
            else:
                indexes = np.arange(len(data))
            if self.top_n is not None:
                indexes = np.argsort(data)[::-1][:self.top_n]
                labels = np.take(labels, indexes)
                data = np.take(data, indexes)
            if self.return_mode == "dict":
                return dict(zip(labels, data))
            else:
                return list(zip(labels, data))
        elif isinstance(data, Iterable):
            return [self._process(elem) for elem in data]
        else:
            raise TypeError(f"Incorrect argument {data}")

    def __call__(self, data):
        return self._process(data)