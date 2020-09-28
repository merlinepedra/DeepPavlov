import sys
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional 

from string import punctuation
PUNCTUATION = punctuation + "«»"

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.dataset_readers.morphotagging_dataset_reader import read_infile as read_ud_infile
from deeppavlov.models.tokenizers.lazy_tokenizer import LazyTokenizer


NO_PUNCT = "NONE"

def make_word_punct_sents(sent, max_punct_in_row=3, remove_initial_punctuation=True,
                          to_remove_punctuation=True):
    word_sent, word_indexes, punct_sent, last_punct = [], [], [], NO_PUNCT
    if not remove_initial_punctuation:
        word_sent.append("")
    for i, word in enumerate(sent):
        if word in PUNCTUATION:
            if last_punct == NO_PUNCT:
                last_punct = word
            elif max_punct_in_row > 0 and len(last_punct) < 2 * max_punct_in_row-1:
                if last_punct.endswith(".") and all(x == "." for x in word):
                    last_punct += word
                else:
                    last_punct += " " + word
            if not to_remove_punctuation:
                word_sent.append(word)
        else:
            if len(word_indexes) > 0 or not remove_initial_punctuation:
                punct_sent.append(last_punct)
            if not to_remove_punctuation and last_punct != NO_PUNCT:
                punct_number = last_punct.count(" ") + 1
                punct_sent.extend(["NONE"] * punct_number) 
            word_sent.append(word)
            word_indexes.append(i)
            last_punct = NO_PUNCT
    if len(word_sent) > 0:
        punct_sent.append(last_punct)
    if not to_remove_punctuation and last_punct != NO_PUNCT:
        punct_number = last_punct.count(" ") + 1
        punct_sent.extend(["NONE"] * punct_number)
    assert len(word_sent) == len(punct_sent)
    return word_sent, punct_sent, word_indexes


def recalculate_head_indexes(heads, word_indexes):
    reverse_word_indexes = {index: i for i, index in enumerate(word_indexes)}
    answer = []
    for new_word_index, word_index in enumerate(word_indexes):
        head_index, new_head_index = word_index, None
        while new_head_index is None:
            head_index = heads[head_index] - 1
            if head_index >= 0:
                new_head_index = reverse_word_indexes.get(head_index)
            else:
                new_head_index = -1
        answer.append(new_head_index+1)    
    return answer


def read_ud_punctuation_file(infile, read_syntax=False, min_length=5, max_sents=-1):
    data = read_ud_infile(infile, read_syntax=read_syntax, max_sents=max_sents)
    word_sents, punct_sents = [], []
    for source_sent, tag_sent in data:
        word_sent, punct_sent, word_indexes = make_word_punct_sents(source_sent)
        if len(word_sent) < min_length:
            continue
        word_sents.append(word_sent)
        if read_syntax:
            tag_sent, head_sent, dep_sent = tag_sent
            tag_sent = [tag_sent[i] for i in word_indexes]
            head_sent = recalculate_head_indexes(head_sent, word_indexes)
            dep_sent = [dep_sent[i] for i in word_indexes]
            punct_sent = (punct_sent, tag_sent, head_sent, dep_sent)
        punct_sents.append(punct_sent)
    return list(zip(word_sents, punct_sents))


def read_punctuation_file(infile, to_tokenize=False, tokenizer=None,
                          max_sents=-1, min_length=5, max_length=60,
                          remove_initial_punctuation=True,
                          to_remove_punctuation=True):
    lines = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line != "":
                # if "..." in line:
                #     continue
                lines.append(line)
                if max_sents != -1 and len(lines) >= max_sents:
                    break
    if to_tokenize:
        tokenizer = tokenizer or LazyTokenizer()
        sents = tokenizer(lines)
    else:
        sents = [sent.split() for sent in lines]
    answer = []
    for sent in sents:
        word_sent, punct_sent, word_indexes = make_word_punct_sents(
            sent, remove_initial_punctuation=remove_initial_punctuation, 
            to_remove_punctuation=to_remove_punctuation)
        if len(word_sent) < min_length:
            continue
        if max_length is not None and len(word_sent) > max_length:
            continue
        answer.append((word_sent, punct_sent))
    return answer


def combine_punctuation_sents(source_words, target_words, target_labels):
    target_index, answer = 0, []
    # while source_index < len(source_words) and target_index < len(target_words):
    for source_index, source_word in enumerate(source_words):
        if all(x in PUNCTUATION for x in source_word):
            answer.append(NO_PUNCT)
            continue
        if target_index >= len(target_words):
            return None
        target_word = target_words[target_index]
        if source_word != target_word:
            return None
        answer.append(target_labels[target_index])
        target_index += 1
    if target_index < len(target_words):
        return None
    return answer

def read_corrupted_punctuation_file(source_file, target_file=None, 
                                    max_sents=-1, min_length=5, max_length=60):
    source_lines, target_lines = [], []
    if target_file is None:
        return read_punctuation_file(source_file, to_tokenize=False,
                                     max_sents=max_sents, min_length=min_length,
                                     max_length=max_length, to_remove_punctuation=False)
    with open(source_file, "r", encoding="utf8") as fsource,\
             open(target_file, "r", encoding="utf8") as ftarget:
        for first_line, second_line in zip(fsource, ftarget):
            first_line, second_line = first_line.strip(), second_line.strip()
            if first_line == "" or second_line == "":
                continue
            if "..." in first_line or "..." in second_line:
                continue
            source_lines.append(first_line.split())
            target_lines.append(second_line.split())
            if max_sents != -1 and len(source_lines) >= max_sents:
                break
    answer = []
    for source_sent, target_sent in zip(source_lines, target_lines):
        source_words, source_labels, _ = make_word_punct_sents(source_sent, to_remove_punctuation=False)
        target_words, target_labels, _ = make_word_punct_sents(target_sent, to_remove_punctuation=True)
        combined_labels = combine_punctuation_sents(source_words, target_words, target_labels)
        if combined_labels is not None:
            answer.append((source_words, combined_labels))
    return answer
        

@register('ud_punctuation_dataset_reader')
class UDPunctuationDatasetReader(DatasetReader):

    def read(self, data_path: List, data_types: Optional[List[str]] = None,
             data_formats=None, skip_short_test_sentences=True, **kwargs) -> Dict[str, List]:
        """Reads UD dataset from data_path.

        Args:
            data_path: can be either
                1. a directory containing files. The file for data_type 'mode'
                is then data_path / {language}-ud-{mode}.conllu
                2. a list of files, containing the same number of items as data_types
            language: a language to detect filename when it is not given
            data_types: which dataset parts among 'train', 'dev', 'test' are returned

        Returns:
            a dictionary containing dataset fragments (see ``read_infile``) for given data types
        """
        if data_types is None:
            data_types = ["train", "valid", "test"]
        if data_formats is None:
            data_formats = ["ud"] * len(data_path)
        answer = {"train": [], "valid": [], "test": []}
        min_length = kwargs.pop("min_length", 5)
        for data_type, infile, data_format in zip(data_types, data_path, data_formats):
            if data_type not in data_types:
                continue
            if data_type != "train" and not skip_short_test_sentences:
                curr_min_length = 1
            else:
                curr_min_length = min_length
            print(data_type, min_length)
            if data_format == "ud":
                answer[data_type] += read_ud_punctuation_file(infile, min_length=curr_min_length, **kwargs)
            elif data_format == "tokenized":
                answer[data_type] += read_punctuation_file(infile, min_length=min_length, to_tokenize=False, **kwargs)
            elif data_format == "text":
                answer[data_type] += read_punctuation_file(infile, min_length=min_length, to_tokenize=True, **kwargs)
            else:
                continue
        return answer

@register('corrupted_punctuation_dataset_reader')
class CorruptedPunctuationDatasetReader:

    def read(self, data_path: List[List[str]], data_types: Optional[List[str]] = None,
             skip_short_test_sentences=True, **kwargs) -> Dict[str, List]:
        """Reads UD dataset from data_path.

        Args:
            data_path: can be either
                1. a directory containing files. The file for data_type 'mode'
                is then data_path / {language}-ud-{mode}.conllu
                2. a list of files, containing the same number of items as data_types
            language: a language to detect filename when it is not given
            data_types: which dataset parts among 'train', 'dev', 'test' are returned

        Returns:
            a dictionary containing dataset fragments (see ``read_infile``) for given data types
        """
        if data_types is None:
            data_types = ["train", "valid", "test"]
        answer = {"train": [], "valid": [], "test": []}
        min_length = kwargs.pop("min_length", 5)
        data_path = list(zip(data_path[0::2], data_path[1::2]))
        for data_type, infiles in zip(data_types, data_path):
            if len(infiles) != 2:
                raise ValueError("Each item in the list of files should be a pair of corrupted and correct file.")
            if data_type not in data_types:
                continue
            if data_type != "train" and not skip_short_test_sentences:
                curr_min_length = 1
            else:
                curr_min_length = min_length
            answer[data_type] += read_corrupted_punctuation_file(*infiles, min_length=min_length, **kwargs)
        return answer
