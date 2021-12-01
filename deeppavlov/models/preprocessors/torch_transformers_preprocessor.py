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
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
import torch
from typing import Tuple, List, Optional, Union, Dict, Set

import numpy as np
import pymorphy2
from transformers import AutoTokenizer, BertTokenizer
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import zero_pad
from deeppavlov.core.models.component import Component
from deeppavlov.models.preprocessors.mask import Mask

log = getLogger(__name__)


@register('torch_transformers_multiplechoice_preprocessor')
class TorchTransformersMultiplechoicePreprocessor(Component):
    """Tokenize text on subtokens, encode subtokens with their indices, create tokens and segment masks.

    Check details in :func:`bert_dp.preprocessing.convert_examples_to_features` function.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of input features and tokens, or only input features

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of input features and tokens, or only input features
        tokenizer: instance of Bert FullTokenizer

    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def tokenize_mc_examples(self,
                             contexts: List[List[str]],
                             choices: List[List[str]]) -> Dict[str, torch.tensor]:

        num_choices = len(contexts[0])
        batch_size = len(contexts)

        # tokenize examples in groups of `num_choices`
        examples = []
        for context_list, choice_list in zip(contexts, choices):
            for context, choice in zip(context_list, choice_list):
                tokenized_input = self.tokenizer.encode_plus(text=context,
                                                             text_pair=choice,
                                                             return_attention_mask=True,
                                                             add_special_tokens=True,
                                                             truncation=True)

                examples.append(tokenized_input)

        padded_examples = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
        )

        padded_examples = {k: v.view(batch_size, num_choices, -1) for k, v in padded_examples.items()}

        return padded_examples

    def __call__(self, texts_a: List[List[str]], texts_b: List[List[str]] = None) -> Dict[str, torch.tensor]:
        """Tokenize and create masks.

        texts_a and texts_b are separated by [SEP] token

        Args:
            texts_a: list of texts,
            texts_b: list of texts, it could be None, e.g. single sentence classification task

        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
        """

        input_features = self.tokenize_mc_examples(texts_a, texts_b)
        return input_features


@register('torch_transformers_preprocessor')
class TorchTransformersPreprocessor(Component):
    """Tokenize text on subtokens, encode subtokens with their indices, create tokens and segment masks.

    Check details in :func:`bert_dp.preprocessing.convert_examples_to_features` function.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of input features and tokens, or only input features

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of input features and tokens, or only input features
        tokenizer: instance of Bert FullTokenizer

    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, texts_a: List[str], texts_b: Optional[List[str]] = None) -> Union[List[InputFeatures],
                                                                                         Tuple[List[InputFeatures],
                                                                                               List[List[str]]]]:
        """Tokenize and create masks.

        texts_a and texts_b are separated by [SEP] token

        Args:
            texts_a: list of texts,
            texts_b: list of texts, it could be None, e.g. single sentence classification task

        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
        """

        # in case of iterator's strange behaviour
        if isinstance(texts_a, tuple):
            texts_a = list(texts_a)

        input_features = self.tokenizer(text=texts_a,
                                        text_pair=texts_b,
                                        add_special_tokens=True,
                                        max_length=self.max_seq_length,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        truncation=True,
                                        return_tensors='pt')
        return input_features


@register('torch_squad_transformers_preprocessor')
class TorchSquadTransformersPreprocessor(Component):
    """Tokenize text on subtokens, encode subtokens with their indices, create tokens and segment masks.

    Check details in :func:`bert_dp.preprocessing.convert_examples_to_features` function.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of input features and tokens, or only input features

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        return_tokens: whether to return tuple of input features and tokens, or only input features
        tokenizer: instance of Bert FullTokenizer

    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 add_token_type_ids: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        self.add_token_type_ids = add_token_type_ids
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    def __call__(self, texts_a: List[str], texts_b: Optional[List[str]] = None) -> Union[List[InputFeatures],
                                                                                         Tuple[List[InputFeatures],
                                                                                               List[List[str]]]]:
        """Tokenize and create masks.

        texts_a and texts_b are separated by [SEP] token

        Args:
            texts_a: list of texts,
            texts_b: list of texts, it could be None, e.g. single sentence classification task

        Returns:
            batch of :class:`transformers.data.processors.utils.InputFeatures` with subtokens, subtoken ids, \
                subtoken mask, segment mask, or tuple of batch of InputFeatures and Batch of subtokens
        """

        if texts_b is None:
            texts_b = [None] * len(texts_a)

        input_features = []
        tokens = []
        for text_a, text_b in zip(texts_a, texts_b):
            encoded_dict = self.tokenizer.encode_plus(
                text=text_a, text_pair=text_b,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt')

            if 'token_type_ids' not in encoded_dict:
                if self.add_token_type_ids:
                    input_ids = encoded_dict['input_ids']
                    seq_len = input_ids.size(1)
                    sep = torch.where(input_ids == self.tokenizer.sep_token_id)[1][0].item()
                    len_a = min(sep + 1, seq_len)
                    len_b = seq_len - len_a
                    encoded_dict['token_type_ids'] = torch.cat((torch.zeros(1, len_a, dtype=int),
                                                                torch.ones(1, len_b, dtype=int)), dim=1)
                else:
                    encoded_dict['token_type_ids'] = torch.tensor([0])

            curr_features = InputFeatures(input_ids=encoded_dict['input_ids'],
                                          attention_mask=encoded_dict['attention_mask'],
                                          token_type_ids=encoded_dict['token_type_ids'],
                                          label=None)
            input_features.append(curr_features)
            if self.return_tokens:
                tokens.append(self.tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0]))

        if self.return_tokens:
            return input_features, tokens
        else:
            return input_features


@register('adopting_preprocessor')
class AdoptingPreprocessor(Component):
    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 return_sent: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        special_tokens_dict = {'additional_special_tokens': ['<TEXT>', '<NER>', '<FREQ_TOPIC>' '<CITES>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.morph = pymorphy2.MorphAnalyzer()
        self.number = 0
        self.return_sent = return_sent
    
    def __call__(self, text_batch, entities_batch, nouns_batch, nouns_inters_batch, topics_batch,
                       topics_inters_batch, classes_batch, entities_sent_batch = None, cites_batch = None):
        wordpiece_tokens_batch = []
        if topics_batch is None:
            topics_batch = [[] for _ in text_batch]
        if entities_sent_batch is None:
            entities_sent_batch = [[] for _ in text_batch]
        if cites_batch is None:
            cites_batch = [[] for _ in text_batch]
        labels_batch = []
        topic_token_dict_batch = []
        token_dict_batch = []
        found_inters_tokens_batch = []
        label_add_tokens_batch = []
        entity_sent_ind_batch = []
        
        for text, entities, nouns_inters, entities_sent, topics, topics_inters, cites, cls in \
                zip(text_batch, entities_batch, nouns_inters_batch, entities_sent_batch, topics_batch,
                    topics_inters_batch, cites_batch, classes_batch):
            labels_list = []
            labels_list.append(cls)
            entity_sent_ind_list = []
            used_entities = set()
            
            topic_token_dict = {}
            doc_wordpiece_tokens = []
            
            if topics:
                topic_tok_cnt = 0
                doc_wordpiece_tokens.append("<FREQ_TOPIC>")
                labels_list.append(0)
                topic_tok_cnt += 1
                
                freq_topics = []
                rare_topics = []
                topics = list(topics.items())
                topics = sorted(topics, key=lambda x: x[1], reverse=True)
                for topic, score in topics:
                    if score > 0.01:
                        freq_topics.append(topic)
                    else:
                        rare_topics.append(topic)
                
                for freq_topic in freq_topics:
                    word_tokens = self.tokenizer.tokenize(freq_topic)
                    doc_wordpiece_tokens += word_tokens
                    if freq_topic in topics_inters:
                        for _ in word_tokens:
                            labels_list.append(1)
                    else:
                        for _ in word_tokens:
                            labels_list.append(0)
                
                    topic_token_dict[freq_topic] = []
                    for _ in word_tokens:
                        topic_token_dict[freq_topic].append(topic_tok_cnt)
                        topic_tok_cnt += 1
                    
                doc_wordpiece_tokens.append("<FREQ_TOPIC>")
                labels_list.append(0)
                topic_tok_cnt += 1
                
                for rare_topic in rare_topics:
                    word_tokens = self.tokenizer.tokenize(rare_topic)
                    doc_wordpiece_tokens += word_tokens
                    if freq_topic in topics_inters:
                        for _ in word_tokens:
                            labels_list.append(1)
                    else:
                        for _ in word_tokens:
                            labels_list.append(0)
                    
                    topic_token_dict[rare_topic] = []
                    for _ in word_tokens:
                        topic_token_dict[rare_topic].append(topic_tok_cnt)
                        topic_tok_cnt += 1
            
            token_dict = {}
            entity_tok_cnt = 0
            doc_wordpiece_tokens.append("<TEXT>")
            labels_list.append(0)
            entity_tok_cnt += 1
            
            entity_start_pos_list = []
            entity_sent_start_pos_list = []
            entity_end_pos_list = []
            
            text_tokens = re.findall(self.re_tokenizer, text)
            for entity in entities:
                entity_tokens = re.findall(self.re_tokenizer, entity)
                for i in range(len(text_tokens) - len(entity_tokens)):
                    matches = 0
                    for j in range(len(entity_tokens)):
                        if text_tokens[(i + j)] == entity_tokens[j]:
                            matches += 1
                    if matches == len(entity_tokens):
                        entity_start_pos_list.append(i)
                        if entity in entities_sent and entity not in used_entities:
                            entity_sent_start_pos_list.append(i)
                            used_entities.add(entity)
                        entity_end_pos_list.append(i + len(entity_tokens))
            
            found_inters_tokens = []
            entity_inters_pos_list = []
            text_tokens = re.findall(self.re_tokenizer, text)
            for entity in nouns_inters:
                entity_tokens = re.findall(self.re_tokenizer, entity)
                for i in range(len(text_tokens) - len(entity_tokens)):
                    matches = 0
                    for j in range(len(entity_tokens)):
                        if text_tokens[(i + j)].lower() == entity_tokens[j].lower():
                            matches += 1
                        elif text_tokens[(i + j)].lower()[:3] == entity_tokens[j].lower()[:3] and \
                                self.morph.parse(text_tokens[(i + j)].lower())[0].normal_form == self.morph.parse(entity_tokens[j].lower())[0].normal_form:
                            matches += 1
                    if matches == len(entity_tokens):
                        entity_inters_pos_list.append((i, i + len(entity_tokens)))
                        found_inters_tokens.append(text_tokens[i:i+len(entity_tokens)])
            found_inters_tokens_batch.append(found_inters_tokens)
            
            label_add_tokens = []
            for i in range(len(text_tokens)):
                if i in entity_start_pos_list:
                    doc_wordpiece_tokens.append("<NER>")
                    labels_list.append(0)
                    entity_tok_cnt += 1
                elif i in entity_end_pos_list:
                    doc_wordpiece_tokens.append("<NER>")
                    labels_list.append(0)
                    entity_tok_cnt += 1
                if i in entity_sent_start_pos_list and len(doc_wordpiece_tokens) < 485:
                    entity_sent_ind_list.append(len(doc_wordpiece_tokens))
                word_tokens = self.tokenizer.tokenize(text_tokens[i])
                found_entity_inters = False
                for entity_inters_pos in entity_inters_pos_list:
                    if i >= entity_inters_pos[0] and i < entity_inters_pos[1]:
                        found_entity_inters = True
                        break
                if found_entity_inters:
                    for _ in word_tokens:
                        labels_list.append(1)
                    label_add_tokens.append(word_tokens)
                else:
                    for _ in word_tokens:
                        labels_list.append(0)
                
                doc_wordpiece_tokens += word_tokens
                
                token_dict[text_tokens[i]] = []
                for _ in word_tokens:
                    token_dict[text_tokens[i]].append(entity_tok_cnt)
                    entity_tok_cnt += 1
                    
            label_add_tokens_batch.append(label_add_tokens)
            
            doc_wordpiece_tokens.append("<TEXT>")
            labels_list.append(0)
            
            if cites:
                doc_wordpiece_tokens.append("<CITES>")
                cites_str = ", ".join(cites)
                word_tokens = self.tokenizer.tokenize(cites_str)
                doc_wordpiece_tokens += word_tokens
                doc_wordpiece_tokens.append("<CITES>")
            
            wordpiece_tokens_batch.append(doc_wordpiece_tokens)
            labels_batch.append(labels_list[:490])
            token_dict_batch.append(token_dict)
            topic_token_dict_batch.append(topic_token_dict)
            entity_sent_ind_batch.append(entity_sent_ind_list)
        
        max_len = max([len(elem) for elem in wordpiece_tokens_batch]) + 2
        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []
        for text, entities, wordpiece_tokens, labels_list, nouns_inters, found_inters_tokens, label_add_tokens in \
                zip(text_batch, entities_batch, wordpiece_tokens_batch, labels_batch, nouns_inters_batch,
                    found_inters_tokens_batch, label_add_tokens_batch):
            encoding = self.tokenizer.encode_plus(wordpiece_tokens, add_special_tokens = True,
                                                  truncation = True, max_length=max_len,
                                                  pad_to_max_length=True, return_attention_mask = True)
            if self.number < 20:
                labeled_toks = []
                for tok, label in zip(wordpiece_tokens[18:], labels_list[19:]):
                    if label == 1:
                        labeled_toks.append(tok)
                self.number += 1
            input_ids_batch.append(encoding["input_ids"][:490])
            attention_mask_batch.append(encoding["attention_mask"][:490])
            token_type_ids_batch.append(encoding["token_type_ids"][:490])
            
        max_len = min(max_len, 490)
        for i in range(len(labels_batch)):
            if len(labels_batch[i]) < max_len:
                for j in range(max_len - len(labels_batch[i])):
                    labels_batch[i].append(0)
            
        text_features = {"input_ids": input_ids_batch,
                         "attention_mask": attention_mask_batch,
                         "token_type_ids": token_type_ids_batch}
            
        if self.return_sent:
            return text_features, labels_batch, topic_token_dict_batch, token_dict_batch, entity_sent_ind_batch
        else:
            return text_features, labels_batch, topic_token_dict_batch, token_dict_batch


@register('adopting_ind_preprocessor')
class AdoptingIndPreprocessor(Component):
    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 return_sent: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        special_tokens_dict = {'additional_special_tokens': ['<text>', '<ner>', '</ner>', '<freq_topic>', '</freq_topic>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.morph = pymorphy2.MorphAnalyzer()
        self.number = 0
        self.return_sent = return_sent
    
    def __call__(self, text_batch, entities_batch, nouns_batch, nouns_inters_batch, topics_batch,
                       topics_inters_batch, classes_batch, entities_sent_batch = None):
        tm_st = time.time()
        wordpiece_tokens_batch = []
        if topics_batch is None:
            topics_batch = [[] for _ in text_batch]
        if entities_sent_batch is None:
            entities_sent_batch = [[] for _ in text_batch]
        
        cls_labels = []
        topic_token_dict_batch = []
        token_dict_batch = []
        found_inters_tokens_batch = []
        label_add_tokens_batch = []
        entity_sent_ind_batch = []
        
        topic_ind_batch, topic_labels_batch = [], []
        token_ind_batch, token_labels_batch = [], []
        
        for text, entities, nouns, nouns_inters, entities_sent, topics, topics_inters, cls in \
                zip(text_batch, entities_batch, nouns_batch, nouns_inters_batch, entities_sent_batch, topics_batch,
                    topics_inters_batch, classes_batch):
            tm1 = time.time()
            cls_labels.append(cls)
            ind = 1
            entity_sent_ind_list = []
            used_entities = set()
            
            topic_token_dict = {}
            doc_wordpiece_tokens = []
            
            pos_topic_ind, neg_topic_ind = [], []
            
            if topics:
                sp_tok_ind = []
                
                topic_tok_cnt = 0
                doc_wordpiece_tokens.append("<freq_topic>")
                sp_tok_ind.append(ind)
                ind += 1
                topic_tok_cnt += 1
                
                freq_topics = []
                rare_topics = []
                topics = list(topics.items())
                topics = sorted(topics, key=lambda x: x[1], reverse=True)
                for topic, score in topics:
                    if score > 0.05 and len(freq_topics) < 3:
                        freq_topics.append(topic)
                    else:
                        rare_topics.append(topic)
                
                for freq_topic in freq_topics:
                    word_tokens = self.tokenizer.tokenize(freq_topic)
                    doc_wordpiece_tokens += word_tokens
                    if freq_topic in topics_inters:
                        for _ in word_tokens:
                            pos_topic_ind.append(ind)
                            ind += 1
                    else:
                        ind += len(word_tokens)
                
                    topic_token_dict[freq_topic] = []
                    for _ in word_tokens:
                        topic_token_dict[freq_topic].append(topic_tok_cnt)
                        topic_tok_cnt += 1
                    
                doc_wordpiece_tokens.append("</freq_topic>")
                sp_tok_ind.append(ind)
                ind += 1
                topic_tok_cnt += 1
                
                for rare_topic in rare_topics:
                    word_tokens = self.tokenizer.tokenize(rare_topic)
                    doc_wordpiece_tokens += word_tokens
                    if rare_topic in topics_inters:
                        for _ in word_tokens:
                            pos_topic_ind.append(ind)
                            ind += 1
                    else:
                        ind += len(word_tokens)
                    
                    topic_token_dict[rare_topic] = []
                    for _ in word_tokens:
                        topic_token_dict[rare_topic].append(topic_tok_cnt)
                        topic_tok_cnt += 1
                        
                for _ in range(len(pos_topic_ind)):
                    while True:
                        neg_ind = random.randint(2, 17)
                        if neg_ind not in pos_topic_ind and neg_ind not in sp_tok_ind:
                            neg_topic_ind.append(neg_ind)
                            break
            
            tm2 = time.time()
            
            token_dict = {}
            entity_tok_cnt = 0
            doc_wordpiece_tokens.append("<text>")
            ind += 1
            entity_tok_cnt += 1
            
            entity_start_pos_list = []
            entity_sent_start_pos_list = []
            entity_end_pos_list = []
            
            noun_inters_pos_list = []
            
            pos_token_ind, neg_token_ind = [], []
            
            text_tokens = re.findall(self.re_tokenizer, text)
            for entity in entities:
                entity_tokens = re.findall(self.re_tokenizer, entity)
                for i in range(len(text_tokens) - len(entity_tokens)):
                    matches = 0
                    for j in range(len(entity_tokens)):
                        if text_tokens[(i + j)].lower() == entity_tokens[j].lower():
                            matches += 1
                    if matches == len(entity_tokens):
                        entity_start_pos_list.append(i)
                        if entity in entities_sent and entity not in used_entities:
                            entity_sent_start_pos_list.append(i)
                            used_entities.add(entity)
                        entity_end_pos_list.append(i + len(entity_tokens))
            
            text_tokens = re.findall(self.re_tokenizer, text)
            for noun in nouns:
                noun_tokens = re.findall(self.re_tokenizer, noun)
                for i in range(len(text_tokens) - len(noun_tokens)):
                    matches = 0
                    for j in range(len(noun_tokens)):
                        if text_tokens[(i + j)].lower() == noun_tokens[j].lower():
                            matches += 1
                    if matches == len(noun_tokens):
                        noun_inters_pos_list.append((i, i + len(noun_tokens)))
            
            tm3 = time.time()
            
            found_inters_tokens = []
            entity_inters_pos_list = []
            text_tokens = re.findall(self.re_tokenizer, text)
            for entity in nouns_inters:
                entity_tokens = re.findall(self.re_tokenizer, entity)
                for i in range(len(text_tokens) - len(entity_tokens)):
                    matches = 0
                    for j in range(len(entity_tokens)):
                        if text_tokens[(i + j)].lower() == entity_tokens[j].lower():
                            matches += 1
                        elif text_tokens[(i + j)].lower()[:3] == entity_tokens[j].lower()[:3] and \
                                self.morph.parse(text_tokens[(i + j)].lower())[0].normal_form == self.morph.parse(entity_tokens[j].lower())[0].normal_form:
                            matches += 1
                    if matches == len(entity_tokens):
                        entity_inters_pos_list.append((i, i + len(entity_tokens)))
                        found_inters_tokens.append(text_tokens[i:i+len(entity_tokens)])
            found_inters_tokens_batch.append(found_inters_tokens)
            
            tm4 = time.time()
            
            label_add_tokens = []
            for i in range(len(text_tokens)):
                if i in entity_start_pos_list:
                    doc_wordpiece_tokens.append("<ner>")
                    ind += 1
                    entity_tok_cnt += 1
                elif i in entity_end_pos_list:
                    doc_wordpiece_tokens.append("</ner>")
                    ind += 1
                    entity_tok_cnt += 1
                if i in entity_sent_start_pos_list and len(doc_wordpiece_tokens) < 485:
                    entity_sent_ind_list.append(len(doc_wordpiece_tokens))
                word_tokens = self.tokenizer.tokenize(text_tokens[i])
                found_entity_inters = False
                for entity_inters_pos in entity_inters_pos_list:
                    if i >= entity_inters_pos[0] and i < entity_inters_pos[1]:
                        found_entity_inters = True
                        break
                
                found_entity_not_inters = False
                for entity_not_inters_pos in noun_inters_pos_list:
                    if i >= entity_not_inters_pos[0] and i < entity_not_inters_pos[1]:
                        found_entity_not_inters = True
                        break
                if found_entity_inters:
                    for _ in word_tokens:
                        pos_token_ind.append(ind)
                        ind += 1
                    label_add_tokens.append(word_tokens)
                elif found_entity_not_inters:
                    for _ in word_tokens:
                        neg_token_ind.append(ind)
                        ind += 1
                else:
                    ind += len(word_tokens)
                
                doc_wordpiece_tokens += word_tokens
                
                token_dict[text_tokens[i]] = token_dict.get(text_tokens[i], [])
                for _ in word_tokens:
                    token_dict[text_tokens[i]].append(entity_tok_cnt)
                    entity_tok_cnt += 1
              
            tm5 = time.time()
            
            pos_token_ind = sorted(list(set(pos_token_ind)))
            neg_token_ind = sorted(list(set(neg_token_ind)))
            if len(pos_token_ind) > len(neg_token_ind):
                pos_token_ind = pos_token_ind[:len(neg_token_ind)]
            else:
                neg_token_ind = neg_token_ind[:len(pos_token_ind)]
                    
            label_add_tokens_batch.append(label_add_tokens)
            doc_wordpiece_tokens.append("<text>")
            
            wordpiece_tokens_batch.append(doc_wordpiece_tokens)
            token_dict_batch.append(token_dict)
            topic_token_dict_batch.append(topic_token_dict)
            entity_sent_ind_batch.append(entity_sent_ind_list)
            
            topic_ind_labels = [(ind_t, 1) for ind_t in pos_topic_ind if ind_t < 485] + \
                               [(ind_t, 0) for ind_t in neg_topic_ind if ind_t < 485]
            token_ind_labels = [(ind_t, 1) for ind_t in pos_token_ind if ind_t < 485] + \
                               [(ind_t, 0) for ind_t in neg_token_ind if ind_t < 485]
            topic_ind_labels = sorted(topic_ind_labels, key=lambda x: x[0])
            token_ind_labels = sorted(token_ind_labels, key=lambda x: x[0])
            topic_ind = [elem[0] for elem in topic_ind_labels]
            topic_labels = [elem[1] for elem in topic_ind_labels]
            token_ind = [elem[0] for elem in token_ind_labels]
            token_labels = [elem[1] for elem in token_ind_labels]
            
            topic_ind_batch.append(topic_ind)
            topic_labels_batch.append(topic_labels)
            token_ind_batch.append(token_ind)
            token_labels_batch.append(token_labels)
        
        max_len = max([len(elem) for elem in wordpiece_tokens_batch]) + 2
        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []
            
        for wordpiece_tokens in wordpiece_tokens_batch:
            encoding = self.tokenizer.encode_plus(wordpiece_tokens, add_special_tokens = True,
                                                  truncation = True, max_length=max_len,
                                                  pad_to_max_length=True, return_attention_mask = True)
            input_ids_batch.append(encoding["input_ids"][:490])
            attention_mask_batch.append(encoding["attention_mask"][:490])
            token_type_ids_batch.append(encoding["token_type_ids"][:490])
            
        text_features = {"input_ids": input_ids_batch,
                         "attention_mask": attention_mask_batch,
                         "token_type_ids": token_type_ids_batch}
            
        return cls_labels, text_features, topic_ind_batch, topic_labels_batch, token_ind_batch, token_labels_batch, \
            topic_token_dict_batch, token_dict_batch, entity_sent_ind_batch


@register('adopting_ind_infer_preprocessor')
class AdoptingIndInferPreprocessor(Component):
    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 return_sent: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.do_lower_case = do_lower_case
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        special_tokens_dict = {'additional_special_tokens': ['<text>', '<ner>', '</ner>', '<freq_topic>', '</freq_topic>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.morph = pymorphy2.MorphAnalyzer()
        self.number = 0
        self.return_sent = return_sent
    
    def __call__(self, text_batch, entities_batch, nouns_batch, nouns_inters_batch, topics_batch, entities_sent_batch = None):
        wordpiece_tokens_batch = []
        if topics_batch is None:
            topics_batch = [[] for _ in text_batch]
        if entities_sent_batch is None:
            entities_sent_batch = [[] for _ in text_batch]
        
        topic_token_dict_batch = []
        token_dict_batch = []
        found_inters_tokens_batch = []
        label_add_tokens_batch = []
        entity_sent_ind_batch = []
        
        topic_ind_batch = []
        token_ind_batch = []
        
        for text, entities, nouns, nouns_inters, entities_sent, topics in \
                zip(text_batch, entities_batch, nouns_batch, nouns_inters_batch, entities_sent_batch, topics_batch):
            if self.do_lower_case:
                text = text.lower()
                entities = [entity.lower() for entity in entities]
                nouns = [noun.lower() for noun in nouns]
                nouns_inters = [noun.lower() for noun in nouns_inters]
                entities_sent = [entity.lower() for entity in entities_sent]
            nouns = [noun for noun in nouns if not any([set(noun.split()).intersection(entity.split()) for entity in entities])]
            nouns_inters =  entities + [noun for noun in nouns_inters if noun in nouns]
            ind = 1
            entity_sent_ind_list = []
            used_entities = set()
            
            topic_token_dict = {}
            doc_wordpiece_tokens = []
            
            all_topic_ind = []
            freq_topics, rare_topics = [], []
            
            if topics:
                sp_tok_ind = []
                
                topic_tok_cnt = 0
                doc_wordpiece_tokens.append("<freq_topic>")
                sp_tok_ind.append(ind)
                ind += 1
                topic_tok_cnt += 1
                
                topics = list(topics.items())
                topics = sorted(topics, key=lambda x: x[1], reverse=True)
                for topic, score in topics:
                    if score > 0.05 and len(freq_topics) < 3:
                        freq_topics.append(topic)
                    else:
                        rare_topics.append(topic)
                
                for freq_topic in freq_topics:
                    word_tokens = self.tokenizer.tokenize(freq_topic)
                    doc_wordpiece_tokens += word_tokens
                    for _ in word_tokens:
                        all_topic_ind.append(ind)
                        ind += 1
                
                    topic_token_dict[freq_topic] = []
                    for _ in word_tokens:
                        topic_token_dict[freq_topic].append(topic_tok_cnt)
                        topic_tok_cnt += 1
                    
                doc_wordpiece_tokens.append("</freq_topic>")
                sp_tok_ind.append(ind)
                ind += 1
                topic_tok_cnt += 1
                
                for rare_topic in rare_topics:
                    word_tokens = self.tokenizer.tokenize(rare_topic)
                    doc_wordpiece_tokens += word_tokens
                    for _ in word_tokens:
                        all_topic_ind.append(ind)
                        ind += 1
                    
                    topic_token_dict[rare_topic] = []
                    for _ in word_tokens:
                        topic_token_dict[rare_topic].append(topic_tok_cnt)
                        topic_tok_cnt += 1
            
            token_dict = {}
            entity_tok_cnt = 0
            doc_wordpiece_tokens.append("<text>")
            ind += 1
            entity_tok_cnt += 1
            
            entity_start_pos_list = []
            entity_sent_start_pos_list = []
            entity_end_pos_list = []
            
            noun_inters_pos_list = []
            
            pos_token_ind, neg_token_ind = [], []
            
            text_tokens = re.findall(self.re_tokenizer, text)
            for entity in entities:
                entity_tokens = re.findall(self.re_tokenizer, entity)
                for i in range(len(text_tokens) - len(entity_tokens)):
                    matches = 0
                    for j in range(len(entity_tokens)):
                        if text_tokens[(i + j)].lower() == entity_tokens[j].lower():
                            matches += 1
                    if matches == len(entity_tokens):
                        entity_start_pos_list.append(i)
                        if entity in entities_sent and i not in used_entities:
                            entity_sent_start_pos_list.append(i)
                            used_entities.add(i)
                        entity_end_pos_list.append(i + len(entity_tokens))
            
            text_tokens = re.findall(self.re_tokenizer, text)
            for noun in nouns:
                noun_tokens = re.findall(self.re_tokenizer, noun)
                for i in range(len(text_tokens) - len(noun_tokens)):
                    matches = 0
                    for j in range(len(noun_tokens)):
                        if text_tokens[(i + j)].lower() == noun_tokens[j].lower():
                            matches += 1
                    if matches == len(noun_tokens):
                        noun_inters_pos_list.append((i, i + len(noun_tokens)))
            
            found_inters_tokens = []
            entity_inters_pos_list = []
            text_tokens = re.findall(self.re_tokenizer, text)
            for entity in nouns_inters:
                entity_tokens = re.findall(self.re_tokenizer, entity)
                for i in range(len(text_tokens) - len(entity_tokens)):
                    matches = 0
                    for j in range(len(entity_tokens)):
                        if text_tokens[(i + j)].lower() == entity_tokens[j].lower():
                            matches += 1
                        elif text_tokens[(i + j)].lower()[:3] == entity_tokens[j].lower()[:3] and \
                                self.morph.parse(text_tokens[(i + j)].lower())[0].normal_form == self.morph.parse(entity_tokens[j].lower())[0].normal_form:
                            matches += 1
                    if matches == len(entity_tokens):
                        entity_inters_pos_list.append((i, i + len(entity_tokens)))
                        found_inters_tokens.append(text_tokens[i:i+len(entity_tokens)])
            found_inters_tokens_batch.append(found_inters_tokens)
            
            label_add_tokens = []
            for i in range(len(text_tokens)):
                if i in entity_start_pos_list:
                    doc_wordpiece_tokens.append("<ner>")
                    ind += 1
                    entity_tok_cnt += 1
                elif i in entity_end_pos_list:
                    doc_wordpiece_tokens.append("</ner>")
                    ind += 1
                    entity_tok_cnt += 1
                word_tokens = self.tokenizer.tokenize(text_tokens[i])
                if i in entity_sent_start_pos_list and len(doc_wordpiece_tokens) < 485:
                    for _ in word_tokens:
                        entity_sent_ind_list.append(ind)
                found_entity_inters = False
                for entity_inters_pos in entity_inters_pos_list:
                    if i >= entity_inters_pos[0] and i < entity_inters_pos[1]:
                        found_entity_inters = True
                        break
                
                found_entity_not_inters = False
                for entity_not_inters_pos in noun_inters_pos_list:
                    if i >= entity_not_inters_pos[0] and i < entity_not_inters_pos[1]:
                        found_entity_not_inters = True
                        break
                if found_entity_inters:
                    for _ in word_tokens:
                        pos_token_ind.append(ind)
                        ind += 1
                    label_add_tokens.append(word_tokens)
                elif found_entity_not_inters:
                    for _ in word_tokens:
                        neg_token_ind.append(ind)
                        ind += 1
                else:
                    ind += len(word_tokens)
                
                doc_wordpiece_tokens += word_tokens
                
                token_dict[text_tokens[i]] = token_dict.get(text_tokens[i], [])
                for _ in word_tokens:
                    token_dict[text_tokens[i]].append(entity_tok_cnt)
                    entity_tok_cnt += 1
                
            pos_token_ind = sorted(list(set(pos_token_ind)))
            neg_token_ind = sorted(list(set(neg_token_ind)))
            if len(pos_token_ind) > len(neg_token_ind):
                pos_token_ind = pos_token_ind[:len(neg_token_ind)]
            else:
                neg_token_ind = neg_token_ind[:len(pos_token_ind)]
                    
            label_add_tokens_batch.append(label_add_tokens)
            doc_wordpiece_tokens.append("<text>")
            
            wordpiece_tokens_batch.append(doc_wordpiece_tokens)
            token_dict_batch.append(token_dict)
            topic_token_dict_batch.append(topic_token_dict)
            entity_sent_ind_batch.append(entity_sent_ind_list)
            
            topic_ind_labels = [(ind_t, 1) for ind_t in all_topic_ind if ind_t < 485]
            token_ind_labels = [(ind_t, 1) for ind_t in pos_token_ind if ind_t < 485] + \
                               [(ind_t, 0) for ind_t in neg_token_ind if ind_t < 485]
            topic_ind_labels = sorted(topic_ind_labels, key=lambda x: x[0])
            token_ind_labels = sorted(token_ind_labels, key=lambda x: x[0])
            topic_ind = [elem[0] for elem in topic_ind_labels]
            token_ind = [elem[0] for elem in token_ind_labels]
            
            topic_ind_batch.append(topic_ind)
            token_ind_batch.append(token_ind)
        
        max_len = max([len(elem) for elem in wordpiece_tokens_batch]) + 2
        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []
            
        for wordpiece_tokens in wordpiece_tokens_batch:
            encoding = self.tokenizer.encode_plus(wordpiece_tokens, add_special_tokens = True,
                                                  truncation = True, max_length=max_len,
                                                  pad_to_max_length=True, return_attention_mask = True)
            input_ids_batch.append(encoding["input_ids"][:490])
            attention_mask_batch.append(encoding["attention_mask"][:490])
            token_type_ids_batch.append(encoding["token_type_ids"][:490])
            
        text_features = {"input_ids": input_ids_batch,
                         "attention_mask": attention_mask_batch,
                         "token_type_ids": token_type_ids_batch}
            
        return text_features, topic_ind_batch, token_ind_batch, \
            topic_token_dict_batch, token_dict_batch, entity_sent_ind_batch


@register('adopting_infer_preprocessor')
class AdoptingInferPreprocessor(Component):
    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = True,
                 max_seq_length: int = 512,
                 return_tokens: bool = False,
                 return_sent: bool = False,
                 **kwargs) -> None:
        self.max_seq_length = max_seq_length
        self.return_tokens = return_tokens
        self.re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        special_tokens_dict = {'additional_special_tokens': ['<TEXT>', '<NER>', '<FREQ_TOPIC>' '<CITES>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.morph = pymorphy2.MorphAnalyzer()
        self.number = 0
        self.return_sent = return_sent
    
    def __call__(self, text_batch, entities_batch, nouns_batch, nouns_inters_batch, topics_batch,
                       topics_inters_batch, entities_sent_batch):
        wordpiece_tokens_batch = []
        if topics_batch is None:
            topics_batch = [[] for _ in text_batch]
        if entities_sent_batch is None:
            entities_sent_batch = [[] for _ in text_batch]
        topic_token_dict_batch = []
        token_dict_batch = []
        found_inters_tokens_batch = []
        label_add_tokens_batch = []
        entity_sent_ind_batch = []
        freq_topics_batch = []
        
        for text, entities, nouns_inters, entities_sent, topics, topics_inters in \
                zip(text_batch, entities_batch, nouns_inters_batch, entities_sent_batch, topics_batch,
                    topics_inters_batch):
            entity_sent_ind_list = []
            
            topic_token_dict = {}
            doc_wordpiece_tokens = []
            
            freq_topics = []
            rare_topics = []
            
            if topics:
                topic_tok_cnt = 0
                doc_wordpiece_tokens.append("<FREQ_TOPIC>")
                topic_tok_cnt += 1
                
                topics = list(topics.items())
                topics = sorted(topics, key=lambda x: x[1], reverse=True)
                for topic, score in topics:
                    if score > 0.03:
                        freq_topics.append(topic)
                    else:
                        rare_topics.append(topic)
                
                for freq_topic in freq_topics:
                    word_tokens = self.tokenizer.tokenize(freq_topic)
                    doc_wordpiece_tokens += word_tokens
                
                    topic_token_dict[freq_topic] = []
                    for _ in word_tokens:
                        topic_token_dict[freq_topic].append(topic_tok_cnt)
                        topic_tok_cnt += 1
                    
                doc_wordpiece_tokens.append("<FREQ_TOPIC>")
                topic_tok_cnt += 1
                
                for rare_topic in rare_topics:
                    word_tokens = self.tokenizer.tokenize(rare_topic)
                    doc_wordpiece_tokens += word_tokens
                    
                    topic_token_dict[rare_topic] = []
                    for _ in word_tokens:
                        topic_token_dict[rare_topic].append(topic_tok_cnt)
                        topic_tok_cnt += 1
            
            freq_topics_batch.append(freq_topics)
            
            token_dict = {}
            entity_tok_cnt = 0
            doc_wordpiece_tokens.append("<TEXT>")
            entity_tok_cnt += 1
            
            entity_start_pos_list = []
            entity_sent_start_pos_list = []
            entity_end_pos_list = []
            
            text_tokens = re.findall(self.re_tokenizer, text)
            last_ind = 0
            for num_ent, entity in enumerate(entities):
                entity_tokens = re.findall(self.re_tokenizer, entity)
                for i in range(last_ind, len(text_tokens) - len(entity_tokens)):
                    matches = 0
                    for j in range(len(entity_tokens)):
                        if text_tokens[(i + j)] == entity_tokens[j]:
                            matches += 1
                    if matches == len(entity_tokens):
                        entity_start_pos_list.append(i)
                        if entity in entities_sent:
                            entity_sent_start_pos_list.append(i)
                        last_ind = i
                        entity_end_pos_list.append(i + len(entity_tokens))
                        break
            
            found_inters_tokens = []
            entity_inters_pos_list = []
            text_tokens = re.findall(self.re_tokenizer, text)
            for entity in nouns_inters:
                entity_tokens = re.findall(self.re_tokenizer, entity)
                for i in range(len(text_tokens) - len(entity_tokens)):
                    matches = 0
                    for j in range(len(entity_tokens)):
                        if text_tokens[(i + j)].lower() == entity_tokens[j].lower():
                            matches += 1
                        elif text_tokens[(i + j)].lower()[:3] == entity_tokens[j].lower()[:3] and \
                                self.morph.parse(text_tokens[(i + j)].lower())[0].normal_form == self.morph.parse(entity_tokens[j].lower())[0].normal_form:
                            matches += 1
                    if matches == len(entity_tokens):
                        entity_inters_pos_list.append((i, i + len(entity_tokens)))
                        found_inters_tokens.append(text_tokens[i:i+len(entity_tokens)])
            found_inters_tokens_batch.append(found_inters_tokens)
            
            label_add_tokens = []
            
            for i in range(len(text_tokens)):
                if i in entity_start_pos_list:
                    doc_wordpiece_tokens.append("<NER>")
                    entity_tok_cnt += 1
                elif i in entity_end_pos_list:
                    doc_wordpiece_tokens.append("<NER>")
                    entity_tok_cnt += 1
                if i in entity_sent_start_pos_list and len(doc_wordpiece_tokens) < 485:
                    entity_sent_ind_list.append(len(doc_wordpiece_tokens))
                word_tokens = self.tokenizer.tokenize(text_tokens[i])
                found_entity_inters = False
                for entity_inters_pos in entity_inters_pos_list:
                    if i >= entity_inters_pos[0] and i < entity_inters_pos[1]:
                        found_entity_inters = True
                        break
                if found_entity_inters:
                    label_add_tokens.append(word_tokens)
                
                doc_wordpiece_tokens += word_tokens
                
                token_dict[text_tokens[i]] = []
                for _ in word_tokens:
                    token_dict[text_tokens[i]].append(entity_tok_cnt)
                    entity_tok_cnt += 1
                    
            label_add_tokens_batch.append(label_add_tokens)
            
            doc_wordpiece_tokens.append("<TEXT>")
            
            wordpiece_tokens_batch.append(doc_wordpiece_tokens)
            token_dict_batch.append(token_dict)
            topic_token_dict_batch.append(topic_token_dict)
            entity_sent_ind_batch.append(entity_sent_ind_list)
        
        max_len = max([len(elem) for elem in wordpiece_tokens_batch]) + 2
        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []
        for text, entities, wordpiece_tokens, nouns_inters, found_inters_tokens, label_add_tokens in \
                zip(text_batch, entities_batch, wordpiece_tokens_batch, nouns_inters_batch,
                    found_inters_tokens_batch, label_add_tokens_batch):
            encoding = self.tokenizer.encode_plus(wordpiece_tokens, add_special_tokens = True,
                                                  truncation = True, max_length=max_len,
                                                  pad_to_max_length=True, return_attention_mask = True)
            
            input_ids_batch.append(encoding["input_ids"][:490])
            attention_mask_batch.append(encoding["attention_mask"][:490])
            token_type_ids_batch.append(encoding["token_type_ids"][:490])
            
        text_features = {"input_ids": input_ids_batch,
                         "attention_mask": attention_mask_batch,
                         "token_type_ids": token_type_ids_batch}
        
        return text_features, topic_token_dict_batch, token_dict_batch, entity_sent_ind_batch, freq_topics_batch


@register('copy_define_postprocessor')
class CopyDefinePostprocessor(Component):
    def __init__(self, **kwargs) -> None:
        self.morph = pymorphy2.MorphAnalyzer()
    
    def __call__(self, class_pred_batch, topic_ind_batch, token_ind_batch, topic_token_dict_batch, token_dict_batch, sent_pred_batch):
        model_output_batch = []
        for class_pred, topic_ind_list, token_ind_list, topic_token_dict, token_dict, sent_pred in \
                zip(class_pred_batch, topic_ind_batch, token_ind_batch, topic_token_dict_batch, token_dict_batch, sent_pred_batch):
            topics = []
            nouns = []
            if class_pred == 1:
                for topic_ind in topic_ind_list:
                    for topic, ind_list in topic_token_dict.items():
                        if topic_ind in ind_list:
                            topics.append(topic)
                            break
                for token_ind in token_ind_list:
                    for token, ind_list in token_dict.items():
                        if token_ind in ind_list and self.morph.parse(token)[0].tag.POS == "NOUN":
                            nouns.append(token)
                            break
            model_output = (class_pred, topics, nouns, sent_pred)
            model_output_batch.append(model_output)
        return model_output_batch


@register('copy_define_infer_postprocessor')
class CopyDefineInferPostprocessor(Component):
    def __init__(self, **kwargs) -> None:
        self.morph = pymorphy2.MorphAnalyzer()
    
    def __call__(self, copy_pred_batch, copy_conf_batch, topic_pred_batch, topics_with_probs_batch, token_ind_batch,
                       topic_token_dict_batch, token_dict_batch, sent_pred_batch, freq_topics_batch):
        model_output_batch = []
        for copy_pred, copy_conf, topic_pred_list, topics_with_probs, token_ind_list, topic_token_dict, token_dict, \
                sent_pred, freq_topics in \
                zip(copy_pred_batch, copy_conf_batch, topic_pred_batch, topics_with_probs_batch, token_ind_batch,
                    topic_token_dict_batch, token_dict_batch, sent_pred_batch, freq_topics_batch):
            topics = []
            nouns = []
            if copy_pred == 1:
                probs_list_dict = {}
                probs_dict = {}
                for topic, topic_indices in topic_token_dict.items():
                    for ind in topic_indices:
                        if topic in probs_list_dict:
                            probs_list_dict[topic].append(topic_pred_list[ind - 1])
                        else:
                            probs_list_dict[topic] = [topic_pred_list[ind - 1]]
                
                for topic in probs_list_dict:
                    probs_dict[topic] = sum(probs_list_dict[topic]) / len(probs_list_dict[topic])
                
                probs_dict = list(probs_dict.items())
                probs_dict = sorted(probs_dict, key=lambda x: x[1], reverse=True)
                
                topics = [elem[0] for elem in probs_dict[:len(freq_topics)]]
                topics = [topic for topic in topics if topics_with_probs[topic] > 0.01]
                
                for token_ind in token_ind_list:
                    for token, ind_list in token_dict.items():
                        if token_ind in ind_list and self.morph.parse(token)[0].tag.POS == "NOUN":
                            nouns.append(token)
                            break
            else:
                copy_conf = 1.0 - copy_conf
            model_output = (copy_pred, copy_conf, topics, nouns, sent_pred)
            model_output_batch.append(model_output)
        return model_output_batch


@register('copy_define_ind_infer_postprocessor')
class CopyDefineIndInferPostprocessor(Component):
    def __init__(self, **kwargs) -> None:
        self.morph = pymorphy2.MorphAnalyzer()
    
    def __call__(self, copy_pred_batch, copy_conf_batch, topic_pred_batch, topics_with_probs_batch, token_ind_batch,
                       topic_token_dict_batch, token_dict_batch, sent_pred_batch, entities_ind_sent_batch):
        model_output_batch = []
        for copy_pred, copy_conf, topic_pred_list, topics_with_probs, token_ind_list, topic_token_dict, token_dict, \
                sent_pred, entities_ind_sent in \
                zip(copy_pred_batch, copy_conf_batch, topic_pred_batch, topics_with_probs_batch, token_ind_batch,
                    topic_token_dict_batch, token_dict_batch, sent_pred_batch, entities_ind_sent_batch):
            topics = []
            nouns = []
            sent_list = []
            if copy_pred == 1:
                probs_dict = {}
                for ind in topic_pred_list:
                    for topic, topic_indices in topic_token_dict.items():
                        if ind in topic_indices:
                            probs_dict[topic] = topics_with_probs[topic]
                
                probs_list = list(probs_dict.items())
                probs_list = sorted(probs_list, key=lambda x: x[1], reverse=True)
                
                topics = [elem[0] for elem in probs_list[:3]]
                topics = [topic for topic in topics if topics_with_probs[topic] > 0.01]
                
                for token_ind in token_ind_list:
                    for token, ind_list in token_dict.items():
                        if token_ind in ind_list and self.morph.parse(token)[0].tag.POS == "NOUN":
                            nouns.append(token)
                            break
                for token_ind, token_sent in zip(entities_ind_sent, sent_pred):
                    for token, ind_list in token_dict.items():
                        if token_ind - 19 in ind_list:
                            sent_list.append((token, token_sent))
                sent_list = [(token, token_sent) for token, token_sent in sent_list if token in nouns]
            else:
                copy_conf = 1.0 - copy_conf
            model_output = (copy_pred, copy_conf, topics, nouns, sent_list)
            model_output_batch.append(model_output)
        return model_output_batch


@register('torch_transformers_ner_preprocessor')
class TorchTransformersNerPreprocessor(Component):
    """
    Takes tokens and splits them into bert subtokens, encodes subtokens with their indices.
    Creates a mask of subtokens (one for the first subtoken, zero for the others).

    If tags are provided, calculates tags for subtokens.

    Args:
        vocab_file: path to vocabulary
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: replace token to <unk> if it's length is larger than this
            (defaults to None, which is equal to +infinity)
        token_masking_prob: probability of masking token while training
        provide_subword_tags: output tags for subwords or for words
        subword_mask_mode: subword to select inside word tokens, can be "first" or "last"
            (default="first")

    Attributes:
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        max_subword_length: rmax lenght of a bert subtoken
        tokenizer: instance of Bert FullTokenizer
    """

    def __init__(self,
                 vocab_file: str,
                 do_lower_case: bool = False,
                 max_seq_length: int = 512,
                 max_subword_length: int = None,
                 token_masking_prob: float = 0.0,
                 provide_subword_tags: bool = False,
                 subword_mask_mode: str = "first",
                 **kwargs):
        self._re_tokenizer = re.compile(r"[\w']+|[^\w ]")
        self.provide_subword_tags = provide_subword_tags
        self.mode = kwargs.get('mode')
        self.max_seq_length = max_seq_length
        self.max_subword_length = max_subword_length
        self.subword_mask_mode = subword_mask_mode
        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = AutoTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
        self.token_masking_prob = token_masking_prob

    def __call__(self,
                 tokens: Union[List[List[str]], List[str]],
                 tags: List[List[str]] = None,
                 **kwargs):
        if isinstance(tokens[0], str):
            tokens = [re.findall(self._re_tokenizer, s) for s in tokens]
        subword_tokens, subword_tok_ids, startofword_markers, subword_tags = [], [], [], []
        for i in range(len(tokens)):
            toks = tokens[i]
            ys = ['O'] * len(toks) if tags is None else tags[i]
            assert len(toks) == len(ys), \
                f"toks({len(toks)}) should have the same length as ys({len(ys)})"
            sw_toks, sw_marker, sw_ys = \
                self._ner_bert_tokenize(toks,
                                        ys,
                                        self.tokenizer,
                                        self.max_subword_length,
                                        mode=self.mode,
                                        subword_mask_mode=self.subword_mask_mode,
                                        token_masking_prob=self.token_masking_prob)
            if self.max_seq_length is not None:
                if len(sw_toks) > self.max_seq_length:
                    raise RuntimeError(f"input sequence after bert tokenization"
                                       f" shouldn't exceed {self.max_seq_length} tokens.")
            subword_tokens.append(sw_toks)
            subword_tok_ids.append(self.tokenizer.convert_tokens_to_ids(sw_toks))
            startofword_markers.append(sw_marker)
            subword_tags.append(sw_ys)
            assert len(sw_marker) == len(sw_toks) == len(subword_tok_ids[-1]) == len(sw_ys), \
                f"length of sow_marker({len(sw_marker)}), tokens({len(sw_toks)})," \
                f" token ids({len(subword_tok_ids[-1])}) and ys({len(ys)})" \
                f" for tokens = `{toks}` should match"

        subword_tok_ids = zero_pad(subword_tok_ids, dtype=int, padding=0)
        startofword_markers = zero_pad(startofword_markers, dtype=int, padding=0)
        attention_mask = Mask()(subword_tokens)

        if tags is not None:
            if self.provide_subword_tags:
                return tokens, subword_tokens, subword_tok_ids, \
                       attention_mask, startofword_markers, subword_tags
            else:
                nonmasked_tags = [[t for t in ts if t != 'X'] for ts in tags]
                for swts, swids, swms, ts in zip(subword_tokens,
                                                 subword_tok_ids,
                                                 startofword_markers,
                                                 nonmasked_tags):
                    if (len(swids) != len(swms)) or (len(ts) != sum(swms)):
                        log.warning('Not matching lengths of the tokenization!')
                        log.warning(f'Tokens len: {len(swts)}\n Tokens: {swts}')
                        log.warning(f'Markers len: {len(swms)}, sum: {sum(swms)}')
                        log.warning(f'Masks: {swms}')
                        log.warning(f'Tags len: {len(ts)}\n Tags: {ts}')
                return tokens, subword_tokens, subword_tok_ids, \
                       attention_mask, startofword_markers, nonmasked_tags
        return tokens, subword_tokens, subword_tok_ids, startofword_markers, attention_mask

    @staticmethod
    def _ner_bert_tokenize(tokens: List[str],
                           tags: List[str],
                           tokenizer: AutoTokenizer,
                           max_subword_len: int = None,
                           mode: str = None,
                           subword_mask_mode: str = "first",
                           token_masking_prob: float = None) -> Tuple[List[str], List[int], List[str]]:
        do_masking = (mode == 'train') and (token_masking_prob is not None)
        do_cutting = (max_subword_len is not None)
        tokens_subword = ['[CLS]']
        startofword_markers = [0]
        tags_subword = ['X']
        for token, tag in zip(tokens, tags):
            token_marker = int(tag != 'X')
            subwords = tokenizer.tokenize(token)
            if not subwords or (do_cutting and (len(subwords) > max_subword_len)):
                tokens_subword.append('[UNK]')
                startofword_markers.append(token_marker)
                tags_subword.append(tag)
            else:
                if do_masking and (random.random() < token_masking_prob):
                    tokens_subword.extend(['[MASK]'] * len(subwords))
                else:
                    tokens_subword.extend(subwords)
                if subword_mask_mode == "last":
                    startofword_markers.extend([0] * (len(subwords) - 1) + [token_marker])
                else:
                    startofword_markers.extend([token_marker] + [0] * (len(subwords) - 1))
                tags_subword.extend([tag] + ['X'] * (len(subwords) - 1))

        tokens_subword.append('[SEP]')
        startofword_markers.append(0)
        tags_subword.append('X')
        return tokens_subword, startofword_markers, tags_subword


@register('torch_bert_ranker_preprocessor')
class TorchBertRankerPreprocessor(TorchTransformersPreprocessor):
    """Tokenize text to sub-tokens, encode sub-tokens with their indices, create tokens and segment masks for ranking.

    Builds features for a pair of context with each of the response candidates.
    """

    def __call__(self, batch: List[List[str]]) -> List[List[InputFeatures]]:
        """Tokenize and create masks.

        Args:
            batch: list of elements where the first element represents the batch with contexts
                and the rest of elements represent response candidates batches

        Returns:
            list of feature batches with subtokens, subtoken ids, subtoken mask, segment mask.
        """

        if isinstance(batch[0], str):
            batch = [batch]

        cont_resp_pairs = []
        if len(batch[0]) == 1:
            contexts = batch[0]
            responses_empt = [None] * len(batch)
            cont_resp_pairs.append(zip(contexts, responses_empt))
        else:
            contexts = [el[0] for el in batch]
            for i in range(1, len(batch[0])):
                responses = []
                for el in batch:
                    responses.append(el[i])
                cont_resp_pairs.append(zip(contexts, responses))

        input_features = []

        for s in cont_resp_pairs:
            sub_list_features = []
            for context, response in s:
                encoded_dict = self.tokenizer.encode_plus(
                    text=context, text_pair=response, add_special_tokens=True, max_length=self.max_seq_length,
                    pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')

                curr_features = InputFeatures(input_ids=encoded_dict['input_ids'],
                                              attention_mask=encoded_dict['attention_mask'],
                                              token_type_ids=encoded_dict['token_type_ids'],
                                              label=None)
                sub_list_features.append(curr_features)
            input_features.append(sub_list_features)

        return input_features


@dataclass
class RecordFlatExample:
    """Dataclass to store a flattened ReCoRD example. Contains `probability` for
    a given `entity` candidate, as well as its label.
    """
    index: str
    label: int
    probability: float
    entity: str


@dataclass
class RecordNestedExample:
    """Dataclass to store a nested ReCoRD example. Contains a single predicted entity, as well as
    a list of correct answers.
    """
    index: str
    prediction: str
    answers: List[str]


@register("torch_record_postprocessor")
class TorchRecordPostprocessor:
    """Combines flat classification examples into nested examples. When called returns nested examples
    that weren't previously returned during current iteration over examples.

    Args:
        is_binary: signifies whether the classifier uses binary classification head
    Attributes:
        record_example_accumulator: underling accumulator that transforms flat examples
        total_examples: overall number of flat examples that must be processed during current iteration
    """

    def __init__(self, is_binary: bool = False, *args, **kwargs):
        self.record_example_accumulator: RecordExampleAccumulator = RecordExampleAccumulator()
        self.total_examples: Optional[int, None] = None
        self.is_binary: bool = is_binary

    def __call__(self,
                 idx: List[str],
                 y: List[int],
                 y_pred_probas: np.ndarray,
                 entities: List[str],
                 num_examples: List[int],
                 *args,
                 **kwargs) -> List[RecordNestedExample]:
        """Postprocessor call

        Args:
            idx: list of string indices
            y: list of integer labels
            y_pred_probas: array of predicted probabilities
            num_examples: list of duplicated total numbers of examples

        Returns:
            List[RecordNestedExample]: processed but not previously returned examples (may be empty in some cases)
        """
        if not self.is_binary:
            # if we have outputs for both classes `0` and `1`
            y_pred_probas = y_pred_probas[:, 1]
        if self.total_examples != num_examples[0]:
            # start over if num_examples is different
            # implying that a different split is being evaluated
            self.reset_accumulator()
            self.total_examples = num_examples[0]
        for index, label, probability, entity in zip(idx, y, y_pred_probas, entities):
            self.record_example_accumulator.add_flat_example(index, label, probability, entity)
            self.record_example_accumulator.collect_nested_example(index)
            if self.record_example_accumulator.examples_processed >= self.total_examples:
                # start over if all examples were processed
                self.reset_accumulator()
        return self.record_example_accumulator.return_examples()

    def reset_accumulator(self):
        """Reinitialize the underlying accumulator from scratch
        """
        self.record_example_accumulator = RecordExampleAccumulator()


class RecordExampleAccumulator:
    """ReCoRD example accumulator

    Attributes:
        examples_processed: total number of examples processed so far
        record_counter: number of examples processed for each index
        nested_len: expected number of flat examples for a given index
        flat_examples: stores flat examples
        nested_examples: stores nested examples
        collected_indices: indices of collected nested examples
        returned_indices: indices that have been returned
    """

    def __init__(self):
        self.examples_processed: int = 0
        self.record_counter: Dict[str, int] = defaultdict(lambda: 0)
        self.nested_len: Dict[str, int] = dict()
        self.flat_examples: Dict[str, List[RecordFlatExample]] = defaultdict(lambda: [])
        self.nested_examples: Dict[str, RecordNestedExample] = dict()
        self.collected_indices: Set[str] = set()
        self.returned_indices: Set[str] = set()

    def add_flat_example(self, index: str, label: int, probability: float, entity: str):
        """Add a single flat example to the accumulator

        Args:
            index: example index
            label: example label (`-1` means that label is not available)
            probability: predicted probability
            entity: candidate entity
        """
        self.flat_examples[index].append(RecordFlatExample(index, label, probability, entity))
        if index not in self.nested_len:
            self.nested_len[index] = self.get_expected_len(index)
        self.record_counter[index] += 1
        self.examples_processed += 1

    def ready_to_nest(self, index: str) -> bool:
        """Checks whether all the flat examples for a given index were collected at this point.
        Args:
            index: the index of the candidate nested example
        Returns:
            bool: indicates whether the collected flat examples can be combined into a nested example
        """
        return self.record_counter[index] == self.nested_len[index]

    def collect_nested_example(self, index: str):
        """Combines a list of flat examples denoted by the given index into a single nested example
        provided that all the necessary flat example have been collected by this time.
        Args:
            index: the index of the candidate nested example
        """
        if self.ready_to_nest(index):
            example_list: List[RecordFlatExample] = self.flat_examples[index]
            entities: List[str] = []
            labels: List[int] = []
            probabilities: List[float] = []
            answers: List[str] = []

            for example in example_list:
                entities.append(example.entity)
                labels.append(example.label)
                probabilities.append(example.probability)
                if example.label == 1:
                    answers.append(example.entity)

            prediction_index = np.argmax(probabilities)
            prediction = entities[prediction_index]

            self.nested_examples[index] = RecordNestedExample(index, prediction, answers)
            self.collected_indices.add(index)

    def return_examples(self) -> List[RecordNestedExample]:
        """Determines which nested example were not yet returned during the current evaluation
        cycle and returns them. May return an empty list if there are no new nested examples
        to return yet.
        Returns:
            List[RecordNestedExample]: zero or more nested examples
        """
        indices_to_return: Set[str] = self.collected_indices.difference(self.returned_indices)
        examples_to_return: List[RecordNestedExample] = []
        for index in indices_to_return:
            examples_to_return.append(self.nested_examples[index])
        self.returned_indices.update(indices_to_return)
        return examples_to_return

    @staticmethod
    def get_expected_len(index: str) -> int:
        """
        Calculates the total number of flat examples denoted by the give index
        Args:
            index: the index to calculate the number of examples for
        Returns:
            int: the expected number of examples for this index
        """
        return int(index.split("-")[-1])
