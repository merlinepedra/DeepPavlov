# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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
from typing import Optional, List, Tuple

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register("compose_inputs_hybrid_ranker")
class ComposeInputsHybridRanker(Component):

    def __init__(self,
                 context_depth: int = 3,
                 model_context_depth: int = 3,
                 num_context_turns: int = 10,
                 use_context_for_query: bool = False,
                 use_user_context_only: bool = False,
                 history_includes_last_utterance: Optional[bool] = True,
                 **kwargs):
        self.query_context_depth = context_depth
        self.model_context_depth = model_context_depth
        self.num_context_turns = num_context_turns
        self.use_history = use_context_for_query
        self.use_user_context_for_query = use_user_context_only
        self.history_includes_last_utterance = history_includes_last_utterance

    def __call__(self,
                 utterances_batch: list,
                 history_batch: list,
                 states_batch: Optional[list]=None) -> Tuple[ List[List[str]], List[List[str]] ]:

        query_batch = []
        expanded_context_batch = []

        for i in range(len(utterances_batch)):
            full_context = history_batch[i][-self.num_context_turns + 1:] + [utterances_batch[i]]  \
                if not self.history_includes_last_utterance else history_batch[i][-self.num_context_turns:]
            expanded_context = self._expand_context(full_context, padding="pre", context_depth=self.query_context_depth)

            # logger.debug(f"expanded_context={expanded_context}")
            # search TF-IDF by last utterance only OR using all history
            if self.use_history:
                queries = []
                for i in expanded_context:
                    if len(i) > 0: queries.append(i)
                queries.append(" ".join(expanded_context))
            elif self.use_user_context_for_query:
                queries = []
                for i in expanded_context[::-1][::2][::-1]:
                    if len(i) > 0: queries.append(i)
                queries.append(" ".join(expanded_context[::-1][::2]).strip())
            else:
                queries = expanded_context[:-1]
            # logger.debug(f"queries={queries}")

            # logger.debug("\n\n[START]\nqueries: " + str(queries))   # DEBUG
            query_batch.append(queries)

            model_expanded_context = self._expand_context(full_context, padding="pre", context_depth=self.model_context_depth)
            # logger.debug(f"model_expanded_context={model_expanded_context}")
            # logger.debug("\nquery expand_context:" + str(model_expanded_context))

            # ### Trick: shift of 2 positions to the left ###
            # for j in range(len(model_expanded_context) - 2):
            #     model_expanded_context[j] = model_expanded_context[j + 2]
            # model_expanded_context[-2] = ''
            # model_expanded_context[-1] = ''
            # ###############################################

            # logger.debug("\nmodel exp_context: " + str(model_expanded_context))  # DEBUG
            expanded_context_batch.append(model_expanded_context)

        return query_batch, expanded_context_batch

    def _expand_context(self, context: List[str], padding: str, context_depth: int = 3) -> List[str]:
        """
        Align context length by using pre/post padding of empty sentences up to ``self.num_turns`` sentences
        or by reducing the number of context sentences to ``self.num_turns`` sentences.

        Args:
            context (List[str]): list of raw context sentences
            padding (str): "post" or "pre" context sentences padding

        Returns:
            List[str]: list of ``self.num_turns`` context sentences
        """
        if padding == "post":
            sent_list = context
            res = sent_list + (self.num_context_turns - len(sent_list)) * \
                  [''] if len(sent_list) < self.num_context_turns else sent_list[:self.num_context_turns]
            return res
        elif padding == "pre":
            # context[-(self.num_turns + 1):-1]  because the last item of `context` is always '' (empty string)
            sent_list = context[-self.num_context_turns:]
            if len(sent_list) <= self.num_context_turns:
                tmp = sent_list[:]
                sent_list = [''] * (self.num_context_turns - len(sent_list))
                sent_list.extend(tmp)

            # print('sent_list', sent_list)   # DEBUG
            for i in range(len(sent_list) - context_depth):
                sent_list[i] = ''  # erase context until the desired depth TODO: this is a trick

            return sent_list

"""
---for 
"context_depth": 2,
"model_context_depth": 2,
"use_user_context_only": true,
"use_context_for_query": false,

utterances_histories has len +1 because it consists last utter

---you can see:

ranking_chitchat_2stage| 2020-03-04 11:54:23.660 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', '', '', 's2', 's2']
ranking_chitchat_2stage| 2020-03-04 11:54:23.661 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s2', '    s2']
ranking_chitchat_2stage| 2020-03-04 11:54:23.661 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', '', '', 's2', 's2']
ranking_chitchat_2stage| 2020-03-04 11:54:23 ('172.28.0.8', 54704) - "POST /model HTTP/1.1" 200
ranking_chitchat_2stage| 2020-03-04 11:54:25.44 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', '', '', 's3', 's3']
ranking_chitchat_2stage| 2020-03-04 11:54:25.44 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s3', '    s3']
ranking_chitchat_2stage| 2020-03-04 11:54:25.44 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', '', '', 's3', 's3']
ranking_chitchat_2stage| 2020-03-04 11:54:25 ('172.28.0.8', 54704) - "POST /model HTTP/1.1" 200
ranking_chitchat_2stage| 2020-03-04 11:54:26.901 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', '', '', 's4', 's4']
ranking_chitchat_2stage| 2020-03-04 11:54:26.901 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s4', '    s4']
ranking_chitchat_2stage| 2020-03-04 11:54:26.902 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', '', '', 's4', 's4']
ranking_chitchat_2stage| 2020-03-04 11:54:26 ('172.28.0.8', 54704) - "POST /model HTTP/1.1" 200
ranking_chitchat_2stage| 2020-03-04 11:54:27.845 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', '', '', 's5', 's5']
ranking_chitchat_2stage| 2020-03-04 11:54:27.845 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s5', '    s5']
ranking_chitchat_2stage| 2020-03-04 11:54:27.846 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', '', '', 's5', 's5']
ranking_chitchat_2stage| 2020-03-04 11:54:27 ('172.28.0.8', 54704) - "POST /model HTTP/1.1" 200
ranking_chitchat_2stage| 2020-03-04 11:54:28.849 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', '', '', 's6', 's6']
ranking_chitchat_2stage| 2020-03-04 11:54:28.849 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s6', '    s6']
ranking_chitchat_2stage| 2020-03-04 11:54:28.850 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', '', '', 's6', 's6']



---for 
"context_depth": 4,
"model_context_depth": 4,
"use_user_context_only": true,
"use_context_for_query": false,

utterances_histories has len +1 because it consists last utter

---you can see:

ranking_chitchat_2stage| 2020-03-04 11:58:00.678 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', 's1', 'Sorry, something went wrong', 's2', 's2']
ranking_chitchat_2stage| 2020-03-04 11:58:00.678 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s1', 's2', '   s1 s2']
ranking_chitchat_2stage| 2020-03-04 11:58:00.679 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', 's1', 'Sorry, something went wrong', 's2', 's2']
ranking_chitchat_2stage| 2020-03-04 11:58:00 ('172.28.0.8', 54850) - "POST /model HTTP/1.1" 200
ranking_chitchat_2stage| 2020-03-04 11:58:02.204 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', 's2', '---s2---', 's3', 's3']
ranking_chitchat_2stage| 2020-03-04 11:58:02.204 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s2', 's3', '   s2 s3']
ranking_chitchat_2stage| 2020-03-04 11:58:02.205 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', 's2', '---s2---', 's3', 's3']
ranking_chitchat_2stage| 2020-03-04 11:58:02 ('172.28.0.8', 54850) - "POST /model HTTP/1.1" 200
ranking_chitchat_2stage| 2020-03-04 11:58:03.264 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', 's3', '---s3---', 's4', 's4']
ranking_chitchat_2stage| 2020-03-04 11:58:03.264 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s3', 's4', '   s3 s4']
ranking_chitchat_2stage| 2020-03-04 11:58:03.264 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', 's3', '---s3---', 's4', 's4']
ranking_chitchat_2stage| 2020-03-04 11:58:03 ('172.28.0.8', 54850) - "POST /model HTTP/1.1" 200
ranking_chitchat_2stage| 2020-03-04 11:58:04.48 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', 's4', '---s4---', 's5', 's5']
ranking_chitchat_2stage| 2020-03-04 11:58:04.48 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s4', 's5', '   s4 s5']
ranking_chitchat_2stage| 2020-03-04 11:58:04.48 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', 's4', '---s4---', 's5', 's5']
ranking_chitchat_2stage| 2020-03-04 11:58:04 ('172.28.0.8', 54850) - "POST /model HTTP/1.1" 200
ranking_chitchat_2stage| 2020-03-04 11:58:04.982 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 55: expanded_context=['', '', '', '', '', '', 's5', '---s5---', 's6', 's6']
ranking_chitchat_2stage| 2020-03-04 11:58:04.982 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 69: queries=['s5', 's6', '   s5 s6']
ranking_chitchat_2stage| 2020-03-04 11:58:04.982 DEBUG in 'deeppavlov.models.ranking.compose_inputs_hybrid_ranker'['compose_inputs_hybrid_ranker'] at line 75: model_expanded_context=['', '', '', '', '', '', 's5', '---s5---', 's6', 's6']



"""