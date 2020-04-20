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
import re
from typing import List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.file import load_pickle

log = getLogger(__name__)


@register('template_matcher')
class TemplateMatcher(Component, Serializable):
    """
        This class matches the question with one of the templates
        to extract entity substrings and define which relations
        corresponds to the question
    """

    def __init__(self, load_path: str, templates_filename: str, **kwargs) -> None:
        """

        Args:
            load_path: path to folder with file with templates
            templates_filename: file with templates
            **kwargs:
        """
        super().__init__(save_path=None, load_path=load_path)
        self._templates_filename = templates_filename
        self.load()

    def load(self) -> None:
        self.templates = load_pickle(self.load_path / self._templates_filename)

    def save(self) -> None:
        raise NotImplementedError

    def __call__(self, questions: List[str]) -> Tuple[List[List[str]], List[List[Tuple[str]]], List[str]]:
        entities_batch = []
        relations_batch = []
        query_type_batch = []
        for question in questions:
            question = question.lower()
            question_length = len(question)
            entities = []
            relations = []
            query_type = ""
            min_length = 100
            for template in self.templates:
                template_len = len(template.replace('xxx', ''))
                template_regexp = "([a-zа-я\d\s\.]*)"+template.replace("xxx", "([a-zа-я\d\s\.]+)")
                fnd = re.findall(template_regexp, question)
                if fnd and str(type(fnd[0])) == "<class 'tuple'>":
                    strstart = fnd[0][0]
                    entities_cand = fnd[0][1:]
                    entity_lengths = [len(entity) for entity in entities_cand]

                    if 0 not in entity_lengths:
                        cur_len = sum(entity_lengths)
                        if cur_len < min_length and len(strstart)+template_len + cur_len == question_length:
                            entities = entities_cand
                            relations = self.templates[template][1:]
                            query_type = self.templates[template][0]
                            min_length = cur_len
            entities_batch.append(entities)
            relations_batch.append(relations)
            query_type_batch.append(query_type)
        log.debug(f"entities_from_template: {entities_batch}")
        log.debug(f"relations_from_template: {relations_batch}")
        log.debug(f"query_from_template: {query_type_batch}")

        return entities_batch, relations_batch, query_type_batch

