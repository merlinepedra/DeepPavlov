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
from typing import List, Dict, Tuple, Optional, Any

import nltk
import pymorphy2
from rapidfuzz import fuzz
import sqlite3

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.spelling_correction.levenshtein.levenshtein_searcher import LevenshteinSearcher
from deeppavlov.models.kbqa.wiki_parser import WikiParser

log = getLogger(__name__)


@register('entity_linker')
class EntityLinker(Component):
    def __init__(self, load_path: str,
                 inverted_index_doc_table: Optional[str] = None,
                 entities_list_doc_table: Optional[str] = None,
                 inverted_index_table: str,
                 entities_list_table: str,
                 lemmatize: bool = False,
                 use_prefix_tree: bool = False,
                 link_with_docs: bool = False,
                 **kwargs) -> None:
        
        super().__init__(save_path=None, load_path=load_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.lemmatize = lemmatize
        self.use_prefix_tree = use_prefix_tree
        self.link_with_docs = link_with_docs
        self.what_to_link = "entities"
        
        self.conn = sqlite3.conn(load_path)
        self.cursor = self.conn.cursor()

        if self.link_with_docs:
            self.inverted_index_doc_table = inverted_index_doc_table
            self.entities_list_doc_table = entities_list_doc_table
        self.inverted_index_table = inverted_index_table
        self.entities_list_table = entities_list_table

        if self.use_prefix_tree:
            alphabet = "!#%\&'()+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz½¿ÁÄ" + \
                       "ÅÆÇÉÎÓÖ×ÚßàáâãäåæçèéêëíîïðñòóôöøùúûüýāăąćČčĐėęěĞğĩīİıŁłńňŌōőřŚśşŠšťũūůŵźŻżŽžơưșȚțəʻ" + \
                       "ʿΠΡβγБМавдежикмностъяḤḥṇṬṭầếờợ–‘’Ⅲ−∗"
            dictionary_words = list(self.inverted_index.keys())
            self.searcher = LevenshteinSearcher(alphabet, dictionary_words)

    def __call__(self, entities_batch: str) -> Tuple[List[str], List[float]]:
        self.what_to_link = "entities"
        entity_ids_batch = []
        entity_confidences_batch = []
        for entities in entities_batch:
            entity_ids_list, confidences_list = self.link_entities(entities)
            entity_ids_batch.append(entity_ids_list)
            entity_confidences_batch.append(confidences_list)
        if self.link_with_docs:
            self.what_to_link = "docs"
            doc_ids_batch = []
            doc_confidences_batch = []
            for entities in entities_batch:
                doc_ids_list, confidences_list = self.link_entities(entities)
                doc_ids_batch.append(doc_ids_list)
                doc_confidences_batch.append(confidences_list)
            return entity_ids_batch, entity_confidences_batch, doc_ids_batch, doc_confidences_batch

        else:
            return entity_ids_batch, entity_confidences_batch

    def link_entities(self, entities: List[str]):
        entity_ids_list = []
        confidences_list = []
        for entity in entities:
            candidate_entities = self.candidate_entities_inverted_index(entity)
            entity_ids, confidences = self.sort_found_entities(candidate_entities, entity)
            entity_ids_list.append(entity_ids)
            confidences_list.append(confidences)
        return entity_ids_list, confidences_list

    def candidate_entities_inverted_index(self, entity: str) -> List[Tuple[Any, Any, Any]]:
        word_tokens = nltk.word_tokenize(entity.lower())
        candidate_entities = []

        for tok in word_tokens:
            if len(tok) > 1:
                found = False
                titles_and_popularities = self.extract_title_and_popularity(tok)
                if titles_and_popularities:
                    candidate_entities += titles_and_popularities
                    found = True

                if self.lemmatize:
                    morph_parse_tok = self.morph.parse(tok)[0]
                    lemmatized_tok = morph_parse_tok.normal_form
                    titles_and_popularities = self.extract_title_and_popularity(lemmatized_tok)
                    if titles_and_popularities:
                        candidate_entities += titles_and_popularities
                        found = True

                if not found and self.use_prefix_tree:
                    words_with_levens_1 = self.searcher.search(tok, d=1)
                    for word in words_with_levens_1:
                        candidate_entities += self.extract_title_and_popularity(word[0])

        candidate_entities = list(set(candidate_entities))
        return candidate_entities

    def sort_found_entities(self, candidate_entities: List[Tuple[str]],
                            entity: str) -> Tuple[List[str], List[float], List[Tuple[str, str, int, int]]]:
        entities_ratios = []
        for entity_id, entity_titles, popularity in candidate_entities:
            entity_titles = entity_titles.split('\t')
            fuzz_ratio = max([fuzz.ratio(name.lower(), entity) for name in entity_titles])
            entities_ratios.append((entity_id, fuzz_ratio, popularity))

        srtd_with_ratios = sorted(entities_ratios, key=lambda x: (x[1], x[2]), reverse=True)
        entity_ids = [ent[0] for ent in srtd_with_ratios]
        confidences = [float(ent[1]) * 0.01 for ent in srtd_with_ratios]

        return entity_ids, confidences

    def extract_title_and_popularity(self, word):
        if self.what_to_link == "docs":
            query = f"SELECT e.doc_title, e.doc_title, i.popularity FROM `{self.inverted_index_doc_table}` i" +\
                    f"JOIN `{self.entities_list_doc_table}` e ON i.doc_id = e.doc_id WHERE i.word = '{word}'"
        if self.what_to_link == "entities":
            query = f"SELECT e.entity_qn, e.entity_titles, i.popularity FROM `{self.inverted_index_table}` i" +\
                    f"JOIN `{self.entities_list_table}` e ON i.entity_id = e.entity_id WHERE i.word = '{word}'"
        found_entities = self.cursor.execute()
        return found_entities.fetchall()

