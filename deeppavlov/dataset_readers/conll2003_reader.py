import sys
from logging import getLogger
from pathlib import Path
from typing import Union, List

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress

log = getLogger(__name__)

@register('basic_ner_reader')
class BasicNerReader(DatasetReader):

    def __init__(self, provide_pos: bool = False,
                 provide_doc_ids: bool = False,
                 iob: bool = False,
                 iobes: bool = False,
                 docstart_token: str = None,
                 *args, **kwargs):
        self.provide_pos = provide_pos
        self.provide_doc_ids = provide_doc_ids
        self.iob = iob
        self.iobes = iobes
        # self.sep = sep
        # self.max_length = max_length or sys.maxsize
        self.docstart_token = docstart_token
        self.num_docs = 0
        self.x_is_tuple = self.provide_pos or self.provide_doc_ids
        super(BasicNerReader, self).__init__(*args, **kwargs)

    def read(self, data_path: List[Union[str, Path]],
             data_types: List[str], sep: str = None, 
             max_length=None, **kwargs):
        if len(data_path) != len(data_types):
            raise ValueError("There must be equal number of data types and input files")
        answer = {"train": [], "valid": [], "test": []}    
        for key, infile in zip(data_types, data_path):
            if key == "dev":
                key = "valid"
            answer[key] += self.parse_ner_file(infile, sep=sep, max_length=max_length)
        return answer

    def parse_ner_file(self, file_name: Union[str, Path], sep=None, max_length=None):
        max_length = max_length or sys.maxsize
        if isinstance(file_name, str):
            file_name = Path(file_name)
        samples = []
        with file_name.open(encoding='utf8') as f:
            tokens = []
            pos_tags = []
            tags = []
            for line in f:
                line = line.strip()
                # Check end of the document
                if 'DOCSTART' in line:
                    if len(tokens) > 1:
                        if len(tokens) <= max_length:
                            x = tokens if not self.x_is_tuple else (tokens,)
                            if self.provide_pos:
                                x = x + (pos_tags,)
                            if self.provide_doc_ids:
                                x = x + (self.num_docs,)
                            samples.append((x, tags))
                        tokens = []
                        pos_tags = []
                        tags = []
                    self.num_docs += 1
                    if self.docstart_token is not None:
                        tokens = [self.docstart_token]
                        pos_tags = ['O']
                        tags = ['O']
                elif len(line) < 2:
                    if (len(tokens) > 0) and (tokens != [self.docstart_token]):
                        if len(tokens) <= max_length:
                            x = tokens if not self.x_is_tuple else (tokens,)
                            if self.provide_pos:
                                x = x + (pos_tags,)
                            if self.provide_doc_ids:
                                x = x + (self.num_docs,)
                            samples.append((x, tags))
                        tokens = []
                        pos_tags = []
                        tags = []
                else:
                    if self.provide_pos:
                        try:
                            token, pos, *_, tag = line.split(sep)
                            pos_tags.append(pos)
                        except:
                            log.warning('Skip {}, splitted as {}'.format(repr(line), repr(line.split(sep))))
                            continue
                    else:
                        try:
                            token, *_, tag = line.split(sep)
                        except:
                            log.warning('Skip {}, splitted as {}'.format(repr(line), repr(line.split(sep))))
                            continue

                    tags.append(tag)
                    tokens.append(token)

            if tokens and len(tokens) <= max_length:
                x = tokens if not self.x_is_tuple else (tokens,)
                if self.provide_pos:
                    x = x + (pos_tags,)
                if self.provide_doc_ids:
                    x = x + (self.num_docs,)
                samples.append((x, tags))
                self.num_docs += 1

            if self.iob:
                return [(x, self._iob2_to_iob(tags)) for x, tags in samples]
            if self.iobes:
                return [(x, self._iob2_to_iobes(tags)) for x, tags in samples]
                
        return samples
    
    @staticmethod
    def _iob2_to_iob(tags):
        iob_tags = []

        for n, tag in enumerate(tags):
            if tag.startswith('B-') and (not n or (tags[n - 1][2:] != tag[2:])):
                tag = tag.replace("B-", "I-")
            iob_tags.append(tag)

        return iob_tags

    @staticmethod
    def _iob2_to_iobes(tags):
        tag_map = {"BB": "S", "BO": "S", "IB": "E", "IO": "E"}
        tags = tags + ["O"]
        iobes_tags = []
        for i in range(len(tags) - 1):
            tagtag = tags[i][0] + tags[i + 1][0]
            if tagtag in tag_map:
                iobes_tags.append(tag_map[tagtag] + tags[i][1:])
            else:
                iobes_tags.append(tags[i])
        return iobes_tags


@register('conll2003_reader')
class Conll2003DatasetReader(BasicNerReader):
    """Class to read training datasets in CoNLL-2003 format"""

    def read(self,
             data_path: str,
             dataset_name: str = None,
             provide_pos: bool = False,
             provide_chunk: bool = False,
             provide_doc_ids: bool = False,
             iob: bool = False,
             iobes: bool = False,
             docstart_token: str = None,
             max_length: int = None,
             sep: str = None):
        self.provide_pos = provide_pos
        self.provide_chunk = provide_chunk
        self.provide_doc_ids = provide_doc_ids
        self.iob = iob
        self.iobes = iobes
        self.docstart_token = docstart_token
        self.num_docs = 0
        self.x_is_tuple = self.provide_pos or self.provide_doc_ids
        data_path = Path(data_path)
        files = list(data_path.glob('*.txt'))
        if 'train.txt' not in {file_path.name for file_path in files}:
            if dataset_name == 'conll2003':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/conll2003_v2.tar.gz'
            elif dataset_name == 'collection_rus':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/collection3_v2.tar.gz'
            elif dataset_name == 'ontonotes':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/ontonotes_ner.tar.gz'
            elif dataset_name == 'vlsp2016':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/vlsp2016.tar.gz'
            elif dataset_name == 'dailydialog':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/dailydialog.tar.gz'
            elif dataset_name == 'collection3':
                url = 'http://files.deeppavlov.ai/deeppavlov_data/collection3_anh.tar.gz'
            else:
                raise RuntimeError('train.txt not found in "{}"'.format(data_path))
            data_path.mkdir(exist_ok=True, parents=True)
            download_decompress(url, data_path)
            files = list(data_path.glob('*.txt'))
        dataset = {}

        for file_name in files:
            name = file_name.with_suffix('').name
            dataset[name] = self.parse_ner_file(file_name, max_length=max_length, sep=sep)
        return dataset
