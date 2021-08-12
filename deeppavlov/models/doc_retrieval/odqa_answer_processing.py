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

import re
from io import StringIO
from logging import getLogger
from operator import itemgetter
from typing import List, Union, Tuple, Optional
import pymorphy2
from rusenttokenize import ru_sent_tokenize
from udapi.block.read.conllu import Conllu
from deeppavlov.core.common.registry import register

logger = getLogger(__name__)


@register('odqa_answer_processor')
class ODQAanswerProcessor:
    def __init__(self, syntax_parser, **kwargs):
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.lemmatizer = pymorphy2.MorphAnalyzer()
        self.syntax_parser = syntax_parser
        
    def __call__(self, questions, batch_entity_substr, questions_syntax_info, batch_tags, batch_answers,
                 batch_answer_scores, batch_answer_doc_ids, batch_answer_contexts):
        logger.info(f"batch_entity_substr {batch_entity_substr} batch_tags {batch_tags}")
        selected_answers = []
        selected_answers_conf = []
        for question, question_syntax_info, entity_substr_list, tags, answers, answer_scores, answer_doc_ids, answer_contexts in \
                zip(questions, questions_syntax_info, batch_entity_substr, batch_tags, batch_answers, batch_answer_scores,
                    batch_answer_doc_ids, batch_answer_contexts):
            
            answers, answer_contexts = self.check_paragraphs(question, question_syntax_info, answers, answer_contexts)
            
            answer_counts = {}
            for n, (answer, score) in enumerate(zip(answers, answer_scores)):
                answer_tokens = re.findall(self.re_tokenizer, answer)
                extracted_number = re.findall(f"[\d]{3,4}", answer)
                if extracted_number:
                    answer_lemm = extracted_number[0]
                else:
                    answer_tokens = [tok for tok in answer_tokens
                                     if (len(tok) > 2 or re.findall(f"[\d]+\.[\d]+", tok) \
                                     or re.findall(f"[\d]+,[\d]+", tok) or tok.isdigit())]
                    answer_tokens = [self.lemmatizer.parse(tok)[0].normal_form for tok in answer_tokens]
                    answer_lemm = " ".join(answer_tokens).lower()
                prev_counts = answer_counts.get(answer_lemm, (0, 0, ""))
                answer_counts[answer_lemm] = (prev_counts[0] + 1, prev_counts[1] + (len(answers) - n), answer, score)
            answer_counts = list(answer_counts.items())
            answer_counts = sorted(answer_counts, key=lambda x: (x[1][0], x[1][1]), reverse=True)
            selected_answers.append(answer_counts[0][1][2])
            if answer_counts[0][1][0] >= 2:
                selected_answers_conf.append("HIGH_CONF")
            else:
                selected_answers_conf.append(answer_counts[0][1][3])
            logger.info(f"answer_counts {answer_counts}")
        return selected_answers, selected_answers_conf
    
    def check_paragraphs(self, question, question_syntax_info, answers, answer_contexts):
        filtered_answers = []
        filtered_answer_contexts = []
        q_adj, q_noun, q_place = self.parse_syntax_tree(question_syntax_info)
        if all([q_adj, q_noun, q_place]):
            full_match = []
            for answer, context in zip(answers, answer_contexts):
                context_sentences = ru_sent_tokenize(context)
                context_sent_syntax_info = self.syntax_parser(context_sentences)
                for syntax_info in context_sent_syntax_info:
                    c_adj, c_noun, c_place = self.parse_syntax_tree(syntax_info)
                    if c_adj and c_noun:
                        if q_place in {"мир", "земля", "планета"} and c_place and c_place not in {"мир", "земля", "планета"}:
                            pass
                        else:
                            filtered_answers.append(answer)
                            filtered_answer_contexts.append(context)
                            if (c_adj == q_adj or (c_adj in {"крупный", "большой"} and q_adj in {"крупный", "большой"})) \
                                    and c_noun == q_noun and (c_place == q_place or (c_place in {"мир", "земля", "планета"}
                                                              and q_place in {"мир", "земля", "планета"})):
                                full_match.append((answer, context))
            if full_match:
                filtered_answers = [elem[0] for elem in full_match]
                filtered_answer_contexts = [elem[1] for elem in full_match]
        else:
            filtered_answers = answers
            filtered_answer_contexts = answer_contexts
        
        return filtered_answers, filtered_answer_contexts

    def parse_syntax_tree(self, syntax_tree):
        tree = Conllu(filehandle=StringIO(syntax_tree)).read_tree()
        found_node = ""
        found_supr = ""
        for node in tree.descendants:
            if node.form.lower() in {"самый", "самая", "самое", "самым", "самой", "самые"}:
                found_node = node
                break
        for node in tree.descendants:
            parsed_tok = self.lemmatizer.parse(node.form)[0]
            if parsed_tok.tag.POS == "ADJF" and "Supr" in set(parsed_tok.tag.grammemes):
                found_supr = node
                break
        adj, noun, place = "", "", ""
        if (found_node and self.lemmatizer.parse(found_node.parent.form)[0].tag.POS == "ADJF") or found_supr:
            if found_node:
                adj_node = found_node.parent
                adj = self.lemmatizer.parse(found_node.parent.form)[0].normal_form
            else:
                adj_node = found_supr
                adj = self.lemmatizer.parse(found_supr.form)[0].normal_form
            
            noun_node = ""
            if adj_node.deprel == "root":
                for node in adj_node.children:
                    if node.deprel == "nsubj":
                        noun_node = node
                        noun = node.form
                        break
                for node in adj_node.children:
                    if node.deprel == "nmod":
                        place = node.form
                        break
            elif adj_node.parent.deprel in {"nsubj", "root", "conj", "obl"} \
                    or self.lemmatizer.parse(adj_node.parent.form)[0].tag.POS == "NOUN":
                if adj_node.parent.deprel == "root" and self.lemmatizer.parse(adj_node.parent.form)[0].tag.POS == "VERB":
                    for node in adj_node.parent.children:
                        if node.deprel == "nsubj":
                            noun_node = node
                            noun = node.form
                            break
                    for node in adj_node.children:
                        if node.deprel == "nmod":
                            place = node.form
                            break
                else:
                    noun_node = adj_node.parent
                    noun = adj_node.parent.form
            
            if noun_node and not place:
                for node in noun_node.children:
                    if node.deprel == "nmod":
                        place = node.form
                        break
        if noun:
            noun = self.lemmatizer.parse(noun)[0].normal_form
        if place:
            place = self.lemmatizer.parse(place)[0].normal_form
        
        return adj, noun, place


@register('filter_docs')
class FilterDocs:
    def __init__(self, top_n, **kwargs):
        self.top_n = top_n
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.lemmatizer = pymorphy2.MorphAnalyzer()
        
    def __call__(self, questions, batch_doc_ids, batch_doc_text):
        batch_filtered_doc_ids = []
        batch_filtered_docs = []
        for question, doc_ids, doc_text in zip(questions, batch_doc_ids, batch_doc_text):
            filtered_doc_ids, filtered_docs = self.filter_docs(question, doc_ids, doc_text)
            batch_filtered_doc_ids.append(filtered_doc_ids[:self.top_n])
            batch_filtered_docs.append(filtered_docs[:self.top_n])
        
        return batch_filtered_doc_ids, batch_filtered_docs

    def filter_docs(self, question, doc_ids, docs):
        dist_pattern = re.findall("расстояние от ([\w]+) до ([\w]+)", question)
        filtered_docs = []
        filtered_doc_ids = []
        if dist_pattern:
            places = list(dist_pattern[0])
            lemm_places = []
            lemm_doc_ids = []
            lemm_docs = []
            for place in places:
                place_tokens = re.findall(self.re_tokenizer, place)
                place_tokens = [tok for tok in place_tokens if len(tok) > 2]
                lemm_place = [self.lemmatizer.parse(tok)[0].normal_form for tok in place_tokens]
                lemm_places.append(" ".join(lemm_place))
            print("lemm_places", lemm_places)
            for doc in docs:
                doc_tokens = re.findall(self.re_tokenizer, doc)
                doc_tokens = [tok for tok in doc_tokens if len(tok) > 2]
                lemm_doc = [self.lemmatizer.parse(tok)[0].normal_form for tok in doc_tokens]
                lemm_docs.append(" ".join(lemm_doc))
            for doc_id in doc_ids:
                doc_tokens = re.findall(self.re_tokenizer, doc_id)
                doc_tokens = [tok for tok in doc_tokens if len(tok) > 2]
                lemm_doc = [self.lemmatizer.parse(tok)[0].normal_form for tok in doc_tokens]
                lemm_doc_ids.append(" ".join(lemm_doc))

            for doc_id, doc, lemm_doc_id, lemm_doc in zip(doc_ids, docs, lemm_doc_ids, lemm_docs):
                count = 0
                for place in lemm_places:
                    if place in lemm_doc or place in lemm_doc_id:
                        count += 1
                if count >= len(lemm_places):
                    filtered_docs.append(doc)
                    filtered_doc_ids.append(doc_id)
        else:
            filtered_docs = docs
            filtered_doc_ids = doc_ids

        return filtered_doc_ids, filtered_docs
