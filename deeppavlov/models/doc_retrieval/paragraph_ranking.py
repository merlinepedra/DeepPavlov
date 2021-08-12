import time
from logging import getLogger
import faiss
import numpy as np
import sqlite3
import torch
from transformers import BertTokenizer
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.models.estimator import Component
from deeppavlov.models.ranking.biencoder_long import TextEncoderLong

logger = getLogger(__name__)


@register("paragraph_ranking")
class ParagraphRanking(Component):
    def __init__(self, pretrained_bert,
                       embedder_weights_path,
                       faiss_index_filename,
                       id2doc_title_filename,
                       num_cand_par = 100,
                       nprobe = 10, device = "cpu", **kwargs):
        if device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.pretrained_bert = pretrained_bert
        self.embedder = TextEncoderLong(pretrained_bert = self.pretrained_bert,
                                    bert_tokenizer_config_file = self.pretrained_bert)
        self.embedder_weights_path = embedder_weights_path
        embedder_checkpoint = torch.load(self.embedder_weights_path, map_location=self.device)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
        self.embedder.load_state_dict(embedder_checkpoint["model_state_dict"])
        self.embedder.to(self.device)
        self.faiss_index = faiss.read_index(str(expand_path(faiss_index_filename)))
        self.faiss_index.nprobe = nprobe
        self.id2doc_title = load_pickle(id2doc_title_filename)
        self.num_cand_par = num_cand_par
        
    def __call__(self, questions):
        tm_st = time.time()
        input_text = [[question, None] for question in questions]
        encoding = self.tokenizer.batch_encode_plus(input_text, add_special_tokens = True,
                           pad_to_max_length=True, return_attention_mask = True)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding["token_type_ids"]
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = torch.LongTensor(attention_mask).to(self.device)
        token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
        embs = self.embedder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).detach().cpu().numpy()
        D, I = self.faiss_index.search(embs, self.num_cand_par)
        doc_titles_batch = []
        for scores_list, ind_list in zip(D, I):
            doc_titles_list = []
            for ind in ind_list:
                doc_titles_list.append(self.id2doc_title[ind])
            doc_titles_batch.append(doc_titles_list)
        tm_end = time.time()
        
        return doc_titles_batch
