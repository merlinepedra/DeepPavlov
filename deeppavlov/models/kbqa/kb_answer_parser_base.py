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
from string import punctuation
from typing import List, Tuple, Optional, Dict
import sqlite3

from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.utils import expand_path

log = getLogger(__name__)


class KBBase(Component):
    """Base class to generate an answer for a given question using Wikidata."""

    def __init__(self, load_path: str, *args, **kwargs) -> None:
        self.conn = sqlite3.connect(str(expand_path(load_path)))
        self.cursor = self.conn.cursor()

    def is_kbqa_question(self, question_init: str, lang: str) -> bool:
        is_kbqa = True
        not_kbqa_question_templates_rus = ["почему", "когда будет", "что будет", "что если", "для чего ", "как ",
                                           "что делать", "зачем", "что может"]
        not_kbqa_question_templates_eng = ["why", "what if", "how"]
        kbqa_question_templates_rus = ["как зовут", "как называется", "как звали", "как ты думаешь", "как твое мнение",
                                       "как ты считаешь"]

        question = ''.join([ch for ch in question_init if ch not in punctuation]).lower()
        if lang == "rus":
            is_kbqa = (all(template not in question for template in not_kbqa_question_templates_rus) or
                       any(template in question for template in kbqa_question_templates_rus))
        if lang == "eng":
            is_kbqa = all(template not in question for template in not_kbqa_question_templates_eng)
        return is_kbqa

    def parse_wikidata_object(self,
                              objects_batch: List[str],
                              confidences_batch: List[float]) -> Tuple[List[str], List[float]]:
        parsed_objects = []
        for n, obj in enumerate(objects_batch):
            if len(obj) > 0:
                if obj.startswith('Q'):
                    entity_title = self.q2name(obj)
                    if entity_title:
                        parsed_objects.append(entity_title)
                    else:
                        parsed_objects.append('Not Found')
                        confidences_batch[n] = 0.0
                else:
                    parsed_objects.append(obj)
            else:
                parsed_objects.append('Not Found')
                confidences_batch[n] = 0.0
        return parsed_objects, confidences_batch

    def q2name(self, q_number):
        query = f"SELECT entity_titles FROM entities_list WHERE entity_qn = '{q_number}';"
        query_res = self.cursor.execute(query)
        entity_titles = query_res.fetchall()
        log.debug(f"entity titles {entity_titles}")
        if entity_titles:
            return entity_titles[0][0].split('\t')[0]
        return ""

    def match_triplet(self,
                      entity_triplets: List[List[Tuple[str]]],
                      entity_linking_confidences: List[float],
                      relations: List[str],
                      relation_probs: List[float]) -> Tuple[str, float]:
        obj = ''
        confidence = 0.0
        #log.debug(f"match_triplet, entity_triplets: {entity_triplets[:5]}")
        log.debug(f"match_triplet, relations: {relations}")
        for predicted_relation, rel_prob in zip(relations, relation_probs):
            for entities, linking_confidence in zip(entity_triplets, entity_linking_confidences):
                for rel_triplets in entities:
                    _, relation_from_wiki, object_from_wiki  = rel_triplets
                    if predicted_relation == relation_from_wiki:
                        obj = object_from_wiki
                        confidence = linking_confidence * rel_prob
                        return obj, confidence
        return obj, confidence
