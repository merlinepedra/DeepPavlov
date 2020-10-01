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
from pymorphy2.analyzer import Parse
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

    RARE_FEATURES = ["Fixd", "Litr"]
    SPECIAL_FEATURES = ["Patr", "Surn"]

    def __init__(self, save_path: Optional[str] = None, load_path: Optional[str] = None,
                 transform_lemmas=False, check_proper_nouns=False,
                 rare_grammeme_penalty: float = 1.0, long_lemma_penalty: float = 1.0, 
                 **kwargs) -> None:
        self.transform_lemmas = transform_lemmas
        self.check_proper_nouns = check_proper_nouns
                 
        self.rare_grammeme_penalty = rare_grammeme_penalty
        self.long_lemma_penalty = long_lemma_penalty
        self._reset()
        self.analyzer = MorphAnalyzer()
        self.converter = converters.converter("opencorpora-int", "ud20")
        super().__init__(save_path, load_path, **kwargs)

    def _process_lemma(self, lemma, tag, word, pymorphy_parse=None):
        lemma = lemma.replace("ё", "е")
        _, feats = make_pos_and_tag(tag, sep=",", return_mode="dict")
        if pymorphy_parse is not None and "Surn" in pymorphy_parse.tag:
            # баг с фамилиями
            for elem in pymorphy_parse.lexeme:
                curr_tag, curr_lemma = self.converter(str(elem.tag)), elem.word
                if elem.tag.case == "nomn":
                    _, nom_feats = make_pos_and_tag(curr_tag, sep=" ", return_mode="dict")
                    if feats.get("Gender") == nom_feats.get("Gender") and feats.get("Number") == nom_feats.get("Number"):
                        lemma = curr_lemma
                        break
        for suffix in ["ович", "евич", "овна", "евна"]:
            if word.lower().startswith(lemma + suffix):
                lemma += suffix
                break
            elif word.lower().startswith(lemma[:-1] + suffix):
                lemma = lemma[:-1] + suffix
                break
            elif word.lower().startswith(lemma[:-2] + suffix):
                lemma = lemma[:-2] + suffix
                break
        else:
            if feats.get("Case") == "Nom" and word[0].isupper():
                for suffix in ["ов", "ев"]:
                    if word.lower().startswith(lemma + suffix):
                        lemma += suffix
                        break
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

    def _extract_lemma(self, parse: Parse) -> str:
        special_feats = [x for x in self.SPECIAL_FEATURES if x in parse.tag]
        if len(special_feats) == 0:
            return parse.normal_form
        # here we process surnames and patronyms since PyMorphy lemmatizes them incorrectly
        for other in parse.lexeme:
            tag = other.tag
            if any(x not in tag for x in special_feats):
                continue
            if tag.case == "nomn" and tag.gender == parse.tag.gender and tag.number == "sing":
                return other.word
        return parse.normal_form        

    def _lemmatize(self, word: str, tag: Optional[str] = None) -> str:
        lemma = self.memo.get((word, tag))
        if lemma is not None:
            return lemma
        parses = self.analyzer.parse(word)
        best_lemma, best_distance, best_parse = word, np.inf, None
        for i, parse in enumerate(parses):
            curr_tag = self.converter(str(parse.tag))
            distance = get_tag_distance(tag, curr_tag)
            for feat in self.RARE_FEATURES:
                if feat in parse.tag:
                    distance += self.rare_grammeme_penalty
                    break
            if len(word) == 1 and len(parse.normal_form) > 1:
                distance += self.long_lemma_penalty
            if distance < best_distance:
                best_lemma, best_parse, best_distance = self._extract_lemma(parse), parse, distance
                if distance == 0:
                    break
        # if word.isupper() and len(word) > 1:
        #     best_lemma = best_lemma.upper()
        # elif word[0].isupper():
        #     best_lemma = best_lemma[0].upper() + best_lemma[1:]
        best_lemma = self._process_lemma(best_lemma, tag, word, pymorphy_parse=best_parse)
        self.memo[(word, tag)] = best_lemma
        return best_lemma

NUM_MAPPING = {"пять": "пятый", "четыре": "четвертый", "шесть": "шестой", "десять": "десятый"}

@register("morphorueval_lemmatizer")
class MorphoRuEvalLemmatizer(UDPymorphyLemmatizer):

    LETTERS_TO_REPLACE = {"ѣ": "е", "і": "и", "Ѳ": "Ф", "ꙋ": "у"}
    OLD_MAPPINGS = {"человек": "человѣкъ", "велеть": "велѣти", "лес": "лѣсъ"}

    def __init__(self, is_old=False, *args, **kwargs):
        self.is_old = is_old
        super().__init__(*args, **kwargs)
        
    def _process_lemma(self, lemma, tag, word, pymorphy_parse=None):
        pos, feats = make_pos_and_tag(tag, return_mode="dict")
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
        # сокращения
        elif lemma in ["г.", "гг.", "г", "гг"] and not self.is_old:
            lemma = "год"
        elif word in ["км", "см"]:
            lemma = word
        # местоимения
        elif lemma == "тот" and "Neut" in tag and "PRON" in tag:
            lemma = "то"
        elif lemma in ["основный"]:
            lemma = "основной"
        # %
        elif word == "%":
            lemma = "процент-знак"
        # наречия
        elif word == "надо" and ("ADV" in tag or "VERB" in tag):
            lemma = "надо"
        lemma = UDPymorphyLemmatizer._process_lemma(self, lemma, tag, word, pymorphy_parse=pymorphy_parse)
        if self.is_old:
            if lemma in self.OLD_MAPPINGS:
                return self.OLD_MAPPINGS[lemma]
            if word in ["г", "г."]:
                lemma = "государь"
            if word in ["x."]:
                lemma = "холопъ"
            if word in ["т."]:
                lemma = "твой"
            if pos == "VERB" and lemma[-2:] == "ть":
                lemma = lemma[:-2] + "ти"
                if lemma.endswith("тити") and not word.endswith("тити"):
                    for x in ["платити", "пустити", "чистити"]:
                        if lemma.endswith(x):
                            break
                    else:
                        lemma = lemma[:-2]
            if pos in ["NOUN", "PROPN", "PRON", "DET", "ADP"]:
                if lemma[-1] in "бвгдзклмнпрстфхж":
                    lemma += "ъ"
                elif lemma[-1] in "чшщ" or lemma == "отец":
                    lemma += "ь"
            d = 0
            for i, (x, y) in enumerate(zip(lemma, word)):
                if y != x and y != self.LETTERS_TO_REPLACE.get(x, x):
                    d = i-1
                    break
            else:
                d = min(len(lemma), len(word))
            if d >= 0:
                lemma = word[:d] + lemma[d:]
        if lemma == "" or (word[-1] == "ъ" and lemma[-1] != "ъ"):
            lemma += "ъ"
        return lemma