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
from deeppavlov.core.common.registry import register

logger = getLogger(__name__)


@register('odqa_kbqa_choose')
class ODKBQAChoose:
    def __init__(self, **kwargs):
        pass
        
    def __call__(self, questions, odqa_answers, odqa_answer_scores, kbqa_answers):
        results = []
        results_scores = []
        for question, odqa_answer, odqa_answer_score, kbqa_answer in \
                zip(questions, odqa_answers, odqa_answer_scores, kbqa_answers):
            logger.info(f"question {question} odqa_answer {odqa_answer} odqa_answer_score {odqa_answer_score} kbqa_answer {kbqa_answer}")
            res = odqa_answer
            results.append(res)
            results_scores.append(odqa_answer_score)
            # if any([elem in question.lower() for elem in ["что такое", "кто такой"]):
        return results, results_scores
