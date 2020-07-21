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

"""File containing common operation with keras.backend objects"""
import ujson as json

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

from typing import Union, Optional, Tuple, List

from tensorflow.keras import backend as K
import numpy as np

EPS = 1e-15


# AUXILIARY = ['PAD', 'BEGIN', 'END', 'UNKNOWN']
# AUXILIARY_CODES = PAD, BEGIN, END, UNKNOWN = 0, 1, 2, 3


def to_one_hot(x, k):
    """
    Takes an array of integers and transforms it
    to an array of one-hot encoded vectors
    """
    unit = np.eye(k, dtype=int)
    return unit[x]


def repeat_(x, k):
    tile_factor = [1, k] + [1] * (K.ndim(x) - 1)
    return K.tile(x[:, None, :], tile_factor)


def make_pos_and_tag(tag: str, sep: str = ",",
                     return_mode: Optional[str] = None) -> Tuple[str, Union[str, list, dict, tuple]]:
    """
    Args:
        tag: the part-of-speech tag
        sep: the separator between part-of-speech tag and grammatical features
        return_mode: the type of return value, can be None, list, dict or sorted_items

    Returns:
        the part-of-speech label and grammatical features in required format
    """
    if tag.endswith(" _"):
        tag = tag[:-2]
    if sep in tag:
        pos, tag = tag.split(sep, maxsplit=1)
    else:
        pos, tag = tag, ("_" if return_mode is None else "")
    if return_mode in ["dict", "list", "sorted_items"]:
        tag = tag.split("|") if tag != "" else []
        if return_mode in ["dict", "sorted_items"]:
            tag = dict(tuple(elem.split("=")) for elem in tag)
            if return_mode == "sorted_items":
                tag = tuple(sorted(tag.items()))
    return pos, tag


def make_full_UD_tag(pos: str, tag: Union[str, list, dict, tuple],
                     sep: str = ",", mode: Optional[str] = None) -> str:
    """
    Args:
        pos: the part-of-speech label
        tag: grammatical features in the format, specified by 'mode'
        sep: the separator between part of speech and features in output tag
        mode: the input format of tag, can be None, list, dict or sorted_items

    Returns:
        the string representation of morphological tag
    """
    if tag == "_" or len(tag) == 0:
        return pos
    if mode == "dict":
        tag, mode = sorted(tag.items()), "sorted_items"
    if mode == "sorted_items":
        tag, mode = ["{}={}".format(*elem) for elem in tag], "list"
    if mode == "list":
        tag = "|".join(tag)
    return pos + sep + tag


def _are_equal_pos(first, second):
    NOUNS, VERBS, CONJ = ["NOUN", "PROPN"], ["AUX", "VERB"], ["CCONJ", "SCONJ"]
    return (first == second or any((first in parts) and (second in parts)
                                   for parts in [NOUNS, VERBS, CONJ]))


IDLE_FEATURES = {"Voice", "Animacy", "Degree", "Mood", "VerbForm", "PronType", "Strength"}


def get_tag_distance(first, second, first_sep=",", second_sep=" ",
                     match_similar_pos=True, idle_features=None, pos_diff_weight=2):
    """
    Measures the distance between two (Russian) morphological tags in UD Format.
    The first tag is usually the one predicted by our model (therefore it uses comma
    as separator), while the second is usually the result of automatical conversion,
    where the separator is space.

    Args:
        first: UD morphological tag
        second: UD morphological tag (usually the output of 'russian_tagsets' converter)
        first_sep: separator between two parts of the first tag
        second_sep: separator between two parts of the second tag

    Returns:
        the number of mismatched feature values
    """
    if idle_features is None:
        idle_features = IDLE_FEATURES
    first_pos, first_feats = make_pos_and_tag(first, sep=first_sep, return_mode="dict")
    second_pos, second_feats = make_pos_and_tag(second, sep=second_sep, return_mode="dict")
    if match_similar_pos:
        are_pos_equal = _are_equal_pos(first_pos, second_pos)
    else:
        are_pos_equal = (first_pos == second_pos)
    dist = int(not are_pos_equal) * pos_diff_weight
    for key, value in first_feats.items():
        other = second_feats.get(key)
        if other is None:
            dist += int(key not in idle_features)
        else:
            dist += int(value != other)
    for key in second_feats:
        dist += int(key not in first_feats and key not in idle_features)
    return dist


@register("tag_normalizer")
class TagNormalizer(Estimator):

    def __init__(self, basic_language: Union[int, str] = 0,
                 pos_diff_weight: int = 2, 
                 match_similar_pos: bool = True,
                 idle_features: Optional[List[str]] = None, 
                 save_path: Optional[str] = None, 
                 load_path: Optional[str] = None, 
                 *args, **kwargs):
        self.basic_language = basic_language
        self.pos_diff_weight = pos_diff_weight
        self.match_similar_pos = match_similar_pos
        self.basic_tags = set()
        self.basic_pos = set()
        self.tag_mapping = dict()
        self.idle_features = idle_features if idle_features is not None else IDLE_FEATURES
        self.save_path = save_path
        self.load_path = load_path

    def save(self, *args, **kwargs):
        if self.save_path:
            answer = {"basic_tags": list(self.basic_tags), "basic_pos": list(self.basic_pos),
                      "tag_mapping": self.tag_mapping, "idle_features": self.idle_features}
            with open(self.save_path, "w", encoding="utf8") as fout:
                json.dump(answer, fout, indent=2)

    def load(self, *args, **kwargs):
        if self.load_path:
            with open(self.save_path, "r", encoding="utf8") as fin:
                data = json.load(fin)
            for key, value in data.items():
                if key in ["basic_pos", "basic_tags"]:
                    value = set(value)
                setattr(self, key, value)

    def fit(self, tag_sents, languages):
        for tag_sent, language_index in zip(tag_sents, languages):
            if language_index == self.basic_language:
                for tag in tag_sent:
                    self.basic_tags.add(tag)
                    pos = tag.split(",")[0]
                    self.basic_pos.add(pos)
                    self.tag_mapping[tag] = tag
        for tag_sent, language_index in zip(tag_sents, languages):
            if language_index != self.basic_language:
                for tag in tag_sent:
                    if tag not in self.tag_mapping:
                        self.tag_mapping[tag] = self._find_closest_tag(tag)
        self.save()
        return

    def _find_closest_tag(self, tag):
        if tag in self.tag_mapping:
            return self.tag_mapping[tag]
        pos = tag.split(",")[0]
        if pos not in self.basic_pos:
            return pos
        best_dist, best_tag = np.inf, None
        for basic_tag in self.basic_tags:
            dist = get_tag_distance(basic_tag, tag, first_sep=",", second_sep=",", 
                                    idle_features=self.idle_features,
                                    match_similar_pos=self.match_similar_pos,
                                    pos_diff_weight=self.pos_diff_weight)
            if dist < best_dist:
                best_dist, best_tag = dist, basic_tag
        return best_tag

    def __call__(self, tag_sents, languages=None):
        if languages is None:
            languages = [self.basic_language] * len(tag_sents)
        answer = []
        for tag_sent, language_index in zip(tag_sents, languages):
            curr_answer = []
            for tag in tag_sent:
                if language_index == self.basic_language:
                    basic_tag = tag
                else:
                    basic_tag = self._find_closest_tag(tag)
                self.tag_mapping[tag] = basic_tag
                curr_answer.append(basic_tag)
            answer.append(curr_answer)
        return answer    


@register("hashtag_remover")
class HashtagRemover(Component):

    def __init__(self, *args, **kwargs):
        pass

    def _process(self, word):
        if word[0] not in '#@' or len(word) == 1:
            return word
        return word[1:]

    def __call__(self, batch):
        return [[self._process(word) for word in sent] for sent in batch]


@register("hashtag_adder")
class HashtagAdder(Component):

    def __init__(self, *args, **kwargs):
        pass

    def _process(self, word, lemma):
        if word[0] not in '#@' or len(word) == 1:
            return lemma
        return word[0] + lemma

    def __call__(self, words, lemmas):
        return [[self._process(*elem) for elem in zip(*sent)] for sent in zip(words, lemmas)]

@register("morphorueval_tag_normalizer")
class MorphoRuEvalTagNormalizer(Component):

    def __init__(self, old=False, change_proper_nouns=False, *args, **kwargs):
        self.old = old
        pass

    def _process(self, tag, lemma, word=None):
        pos, feats = make_pos_and_tag(tag, return_mode="dict")
        if self.old:
            if lemma == "прошлое":
                pos = "ADJ"
            if word in ["г.", "т.", "х."]:
                feats["Abbr"] = "Yes"
                if word == "т.":
                    pos = "DET"
                else:
                    pos = "NOUN"
            elif word.endswith(".") and len(word) >= 2 and word[:-1].isalpha():
                pos = "NOUN"
                feats["Abbr"] = "Yes"
        # краткие прилагательные
        if "Variant" in feats and feats["Variant"] == "Short":
            feats["Case"] = "Nom"
        # залог у "быть"
        if lemma == "быть" and pos == "AUX":
            feats.pop("Voice", None)
        # медиальный залог
        if pos == "VERB" and feats.get("VerbForm") != "Part" and feats.get("Voice") == "Pass":
            feats["Voice"] = "Mid"
        # частицы
        if pos == "PART" and lemma in ["не", "ни"]:
            feats["Polarity"] = "Neg"
        # местоимения
        if lemma in ["себя", "себе"]:
            feats["Reflex"] = "Yes"
        # иностранные слова
        if pos == "PROPN" and all("a"<= x <= "z" or x == "-" for x in lemma.lower()):
            pos = "X"
        # наречия
        if lemma == "надо" and pos == "ADV":
            pos = "VERB"
        # 17 век
        if self.old:
            if feats.get("Case", None) != "Acc":
                feats.pop("Animacy", None)
            # if "VerbForm" in feats and feats["VerbForm"] == "Inf":
            #     feats.pop("Voice", None)
            if pos in ["ADJ"] and "Variant" not in feats:
                feats["Variant"] = "Long"
            if pos == "VERB" and feats.get("VerbForm") == "Part" and "Variant" not in feats:
                feats["Variant"] = "Long"
            if pos == "PRON":
                if lemma in ["он", "я", "она", "ты", "мы", "они"]:
                    feats["PronType"] = "Prs"
                elif word == "тебѣ":
                    feats["PronType"] = "Prs"
                elif lemma in ["кто", "который"]:
                    feats["PronType"] = "Rel"
            if pos in ["PRON", "DET"]:
                if lemma == "твой":
                    feats.pop("Variant", None)
                    feats["PronType"] = "Prs"
                if word in ["тѣхъ", "техъ", "сей"] or lemma in ["тѣхъ", "техъ", "тот"]:
                    feats["PronType"] = "Dem"
            if pos == "VERB":
                if feats.get("VerbForm") == "Conv":
                    feats.pop("Aspect", None)
                if feats.get("Tense") == "Past" and feats.get("VerbForm") == "Fin":
                    feats["VerbForm"] = "PartRes"
                    feats.pop("Mood", None)
                if word.endswith("ся"):
                    feats["Reflex"] = "Yes"
            if word == "великихъ":
                feats["Gender"] = "Masc"
            if lemma == "как":
                feats["Degree"] = "Pos"
            if lemma in ["той", "то"]:
                feats["Gender"], feats["PronType"] = "Fem", "Dem"
            if lemma in ["который", "такой"]:
                feats["PronType"] = "Int"
            if lemma in ["никакой"]:
                feats["PronType"] = "Neg"
            if lemma in ["сей"]:
                pos, feats["PronType"] = "DET", "Dem"
            if word in ["тово"]:
                pos, feats["PronType"] = "DET", "Dem"
            if lemma == "весь":
                feats["PronType"] = "Tot"
            if lemma == "денг":
                feats["Gender"] = "Fem"
            if lemma == "они":
                feats["Gender"] = "Masc"
        tag = make_full_UD_tag(pos, feats, mode="dict")
        return tag

    def __call__(self, batch, lemmas, words=None):
        if words is None:
            words = [[None] * len(sent) for sent in lemmas]
        return [[self._process(*elem) for elem in zip(tag_sent, lemmas_sent, words_sent)]
                for tag_sent, lemmas_sent, words_sent in zip(batch, lemmas, words)]
