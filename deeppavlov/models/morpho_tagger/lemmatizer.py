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

from abc import abstractmethod
from typing import List, Optional

import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.morpho_tagger.common_tagger import get_tag_distance, make_full_UD_tag, make_pos_and_tag


class BasicLemmatizer(Serializable):
    """
    A basic class for lemmatizers. It must contain two methods:
    * :meth: `_lemmatize` for single word lemmatization. It is an abstract method and should be reimplemented.
    * :meth: `__call__` for lemmatizing a batch of sentences.
    """

    def __init__(self, save_path: Optional[str] = None,
                 load_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(save_path, load_path, **kwargs)

    @abstractmethod
    def _lemmatize(self, word: str, tag: Optional[str] = None) -> str:
        """
        Lemmatizes a separate word given its tag.

        Args:
            word: the input word.
            tag: optional morphological tag.

        Returns:
            a lemmatized word
        """
        raise NotImplementedError("Your lemmatizer must implement the abstract method _lemmatize.")

    def __call__(self, data: List[List[str]], tags: Optional[List[List[str]]] = None) -> List[List[str]]:
        """
        Lemmatizes each word in a batch of sentences.

        Args:
            data: the batch of sentences (lists of words).
            tags: the batch of morphological tags (if available).

        Returns:
            a batch of lemmatized sentences.
        """
        if tags is None:
            tags = [[None for _ in sent] for sent in data]
        if len(tags) != len(data):
            raise ValueError("There must be the same number of tag sentences as the number of word sentences.")
        if any((len(elem[0]) != len(elem[1])) for elem in zip(data, tags)):
            raise ValueError("Tag sentence must be of the same length as the word sentence.")
        answer = [[self._lemmatize(word, tag) for word, tag in zip(*elem)] for elem in zip(data, tags)]
        return answer


@register("UD_pymorphy_lemmatizer")
class UDPymorphyLemmatizer(BasicLemmatizer):
    """
    A class that returns a normal form of a Russian word given its morphological tag in UD format.
    Lemma is selected from one of PyMorphy parses,
    the parse whose tag resembles the most a known UD tag is chosen.
    """

    def __init__(self, save_path: Optional[str] = None, load_path: Optional[str] = None,
                 transform_lemmas=False, check_proper_nouns=False, **kwargs) -> None:
        self.transform_lemmas = transform_lemmas
        self.check_proper_nouns = check_proper_nouns
        self._reset()
        self.analyzer = MorphAnalyzer()
        self.converter = converters.converter("opencorpora-int", "ud20")
        super().__init__(save_path, load_path, **kwargs)

    def _process_lemma(self, lemma, tag, word):
        if "PROPN" in tag:
            if word.isupper():
                lemma = lemma.upper()
            elif word[0].isupper():
                lemma = lemma[0].upper() + lemma[1:]
        return lemma

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def _reset(self):
        self.memo = dict()

    def _lemmatize(self, word: str, tag: Optional[str] = None) -> str:
        lemma = self.memo.get((word, tag))
        if lemma is not None:
            return lemma
        parses = self.analyzer.parse(word)
        best_lemma, best_distance = word, np.inf
        for i, parse in enumerate(parses):
            curr_tag, curr_lemma = self.converter(str(parse.tag)), parse.normal_form
            distance = get_tag_distance(tag, curr_tag)
            if distance < best_distance:
                best_lemma, best_parse, best_distance = curr_lemma, parse, distance
                if distance == 0:
                    break
        # if word.isupper() and len(word) > 1:
        #     best_lemma = best_lemma.upper()
        # elif word[0].isupper():
        #     best_lemma = best_lemma[0].upper() + best_lemma[1:]
        best_lemma = self._process_lemma(best_lemma, tag, word)
        self.memo[(word, tag)] = best_lemma
        return best_lemma

NUM_MAPPING = {"пять": "пятый", "четыре": "четвертый", "шесть": "шестой", "десять": "десятый"}

@register("morphorueval_lemmatizer")
class MorphoRuEvalLemmatizer(UDPymorphyLemmatizer):

    def _process_lemma(self, lemma, tag, word):
        # числительные
        if lemma == "два" and word.lower().startswith("втор"):
            lemma = "второй"
        elif lemma == "три" and word.lower().startswith("трет"):
            lemma = "третий" 
        elif lemma == "один" and word.lower().startswith("перв"):
            lemma = "первый"
        elif lemma in NUM_MAPPING and "ADJ" in tag:
            lemma = NUM_MAPPING[lemma]
        # предлоги
        elif word in ["об", "во", "со"] and "ADP" in tag:
            lemma = word
        # наречия
        elif lemma in ["основный"]:
            lemma = "основной"
        lemma = UDPymorphyLemmatizer._process_lemma(self, lemma, tag, word)
        return lemma