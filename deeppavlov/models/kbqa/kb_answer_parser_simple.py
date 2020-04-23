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
from typing import List, Tuple, Optional

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.models.kbqa.kb_answer_parser_base import KBBase

log = getLogger(__name__)


@register('kb_answer_parser_simple')
class KBAnswerParserSimple(KBBase):

    def __init__(self, top_k_classes: int,
                 rule_filter_entities: bool = False,
                 language: str = "eng",
                 *args, **kwargs) -> None:
        
        self.top_k_classes = top_k_classes
        self.rule_filter_entities = rule_filter_entities
        self.language = language
        super().__init__(*args, **kwargs)

    def __call__(self, questions_batch: List[str],
                 entity_ids_batch,
                 entity_confs_batch,
                 rels_from_template_batch,
                 rels_from_nn_batch: Optional[List[List[str]]] = None,
                 rels_probs_batch: Optional[List[List[float]]] = None,
                 *args, **kwargs) -> Tuple[List[str], List[float]]:

        objects_batch = []
        confidences_batch = []

        if rels_from_nn_batch == None:
            rels_from_nn_batch = [[] for i in range(len(questions_batch))]
            rels_probs_batch = [[] for i in range(len(questions_batch))]
        log.debug(f"entity_ids in answer parser {entity_ids_batch}")

        for question, entity_ids_list, entity_confs_list, rels_from_template, rels_from_nn, rels_probs in \
            zip(questions_batch, entity_ids_batch, entity_confs_batch, rels_from_template_batch, rels_from_nn_batch, rels_probs_batch):

            is_kbqa = self.is_kbqa_question(question, self.language)
            if is_kbqa:
                entity_ids = []
                for ids_for_entity in entity_ids_list:
                    entity_ids += ids_for_entity[:10]
                entity_confs = []
                for confs_for_entity in entity_confs_list:
                    entity_confs += confs_for_entity[:10]
                entity_triplets = self.extract_triplets_from_wiki(entity_ids)
                if rels_from_template:
                    predicted_relations = [rels_from_template[0][0]]
                    predicted_rel_probs = [1.0]
                elif rels_from_nn:
                    predicted_relations = rels_from_nn
                    predicted_rel_probs = self._parse_relations_probs(relations_probs)
                else:
                    predicted_relations, predicted_rel_probs = [], []
                
                if self.rule_filter_entities and self.language == 'rus':
                    entity_ids, entity_triplets, entity_confs = \
                        self.filter_triplets_rus(entity_triplets, entity_confs, question, entity_ids)

                relation_prob = 1.0
                obj, confidence = self.match_triplet(entity_triplets,
                                                         entity_confs,
                                                         predicted_relations,
                                                         predicted_rel_probs)
                
                log.debug(f"found object {obj}")
                objects_batch.append(obj)
                confidences_batch.append(confidence)
            else:
                objects_batch.append('')
                confidences_batch.append(0.0)

        parsed_objects_batch, confidences_batch = self.parse_wikidata_object(objects_batch, confidences_batch)

        return parsed_objects_batch, confidences_batch

    def _parse_relations_probs(self, probs: List[float]) -> List[float]:
        top_k_inds = np.asarray(probs).argsort()[-self.top_k_classes:][::-1]
        top_k_probs = [probs[k] for k in top_k_inds]
        return top_k_probs

    def extract_triplets_from_wiki(self, entity_ids: List[str]) -> List[List[List[str]]]:
        entity_triplets = []
        for entity_id in entity_ids:
            if entity_id.startswith('Q'):
                query = f"SELECT subject, relation, object FROM wikidata WHERE subject = '{entity_id}';"
                query_res = self.cursor.execute(query)
                entity_triplets.append(query_res.fetchall())
            else:
                entity_triplets.append([])

        return entity_triplets

    def filter_triplets_rus(self, entity_triplets: List[List[List[str]]], confidences: List[float],
                            question: str, srtd_cand_ent: List[Tuple[str]]) -> \
                            Tuple[List[Tuple[str]], List[List[List[str]]], List[float]]:

        what_template = 'что '
        found_what_template = question.find(what_template) > -1
        filtered_entity_triplets = []
        filtered_entities = []
        filtered_confidences = []
        for wiki_entity, confidence, triplets_for_entity in zip(srtd_cand_ent, confidences, entity_triplets):
            entity_is_human = False
            entity_is_asteroid = False
            entity_is_named = False
            entity_title = wiki_entity
            if entity_title[0].isupper():
                entity_is_named = True
            property_is_instance_of = 'P31'
            id_for_entity_human = 'Q5'
            id_for_entity_asteroid = 'Q3863'
            for triplet in triplets_for_entity:
                subject, relation, obj = triplet
                if relation == property_is_instance_of and obj == id_for_entity_human:
                    entity_is_human = True
                    break
                if relation == property_is_instance_of and obj == id_for_entity_asteroid:
                    entity_is_asteroid = True
                    break
            if found_what_template and (entity_is_human or entity_is_named or entity_is_asteroid):
                continue
            filtered_entity_triplets.append(triplets_for_entity)
            filtered_entities.append(wiki_entity)
            filtered_confidences.append(confidence)

        return filtered_entities, filtered_entity_triplets, filtered_confidences
