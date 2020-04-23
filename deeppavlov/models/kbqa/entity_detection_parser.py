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
from typing import List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('entity_detection_parser')
class EntityDetectionParser(Component):
    """This class parses probabilities of tokens to be a token from the entity substring."""

    def __init__(self, thres_proba: float = 0.86, what_to_parse: str = "probas", **kwargs):
        self.thres_proba = thres_proba
        self.what_to_parse = what_to_parse

    def __call__(self, question_tokens_batch: List[List[str]],
                 token_info_batch: List[List[List[float]]]) -> List[List[str]]:
        """

        Args:
            question_tokens: tokenized questions
            token_probas: list of probabilities of question tokens to belong to
            "B-TAG" (beginning of entity substring), "I-TAG" (inner token of entity substring)
            or "O-TAG" (not an entity token)
        """
        
        log.debug(f"question_tokens: {question_tokens_batch}")
        log.debug(f"token_info: {token_info_batch}")
        entities_batch = []
        for tokens, token_info in zip(question_tokens_batch, token_info_batch):
            if self.what_to_parse == "probas":
                tags = self.tags_from_probas(tokens, token_info)
            else:
                tags = [1 if tag == 'E' else 0 for tag in token_info]

            entities = self.entities_from_tags(tokens, tags)
            entities_batch.append(entities)
        log.debug(f"entity substrings {entities_batch}")
        return entities_batch

    def tags_from_probas(self, tokens, probas):
        tags = []
        for proba in probas:
            if proba[0] <= self.thres_proba:
                tags.append(1)
            if proba[0] > self.thres_proba:
                tags.append(0)
        return tags

    def entities_from_tags(self, tokens, tags):
        entities = []
        entity = []
        replace_tokens = [(' - ', '-'), (" 's", "'s"), (' .', ''), ('{', ''), ('}', ''), ('  ', ' '), ('"', "'")]

        for tok, tag in zip(tokens, tags):
            if tag:
                entity.append(tok)
            elif len(entity) > 0:
                entity = ' '.join(entity)
                for old, new in replace_tokens:
                    entity = entity.replace(old, new)
                entities.append(entity)
                entity = []

        return entities
