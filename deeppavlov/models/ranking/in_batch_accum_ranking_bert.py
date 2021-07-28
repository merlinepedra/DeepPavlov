import itertools
from pathlib import Path
from logging import getLogger
from typing import List, Optional, Dict, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
# from apex import amp

from deeppavlov.core.commands.utils import expand_path
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertModel, BertTokenizer
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.ranking.siamese_el_ranking_bert import BilinearRanking

log = getLogger(__name__)


@register('torch_transformers_accum_ranker')
class BertAccumRanker(TorchModel):

    def __init__(
            self,
            model_name: str,
            question_encoder_save_path: str,
            context_encoder_save_path: str,
            pretrained_bert: str = None,
            bert_config_file: Optional[str] = None,
            criterion: str = "CrossEntropyLoss",
            optimizer: str = "AdamW",
            optimizer_parameters: Dict = {"lr": 1e-5, "weight_decay": 0.01, "eps": 1e-6},
            return_probas: bool = False,
            attention_probs_keep_prob: Optional[float] = None,
            hidden_keep_prob: Optional[float] = None,
            clip_norm: Optional[float] = None,
            threshold: Optional[float] = None,
            **kwargs
    ):
        self.question_encoder_save_path = question_encoder_save_path
        self.context_encoder_save_path = context_encoder_save_path
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            criterion=criterion,
            optimizer_parameters=optimizer_parameters,
            return_probas=return_probas,
            **kwargs)

    def train_on_batch(self, q_features: List[Dict],
                             c_features_list: List[List[Dict]],
                             positive_idx: List[List[int]]) -> float:
        input_list_context = []
        for i in range(len(q_features)):
            _input = {'positive_idx': [0]}
            for elem in ['input_ids', 'attention_mask']:
                inp_elem = [q_features[i][elem]]
                _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
            for elem in ['input_ids', 'attention_mask']:
                inp_elem = [f[elem] for f in c_features_list[i]]
                _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
            input_list_context.append(_input)
        
        input_list_question = []
        q_inp_elem_ids = [f["input_ids"] for f in q_features]
        q_inp_elem_att = [f["attention_mask"] for f in q_features]
        for i in range(len(q_features)):
            _input = {"positive_idx": [i]}
            _input["q_input_ids"] = torch.LongTensor(q_inp_elem_ids).to(self.device)
            _input["q_attention_mask"] = torch.LongTensor(q_inp_elem_att).to(self.device)
            for elem in ['input_ids', 'attention_mask']:
                inp_elem = [c_features_list[i][0][elem]]
                _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
            input_list_question.append(_input)
        
        input_list = []
        for i in range(len(input_list_context)):
            input_list.append(input_list_context[i])
            input_list.append(input_list_question[i])
        
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()      # zero the parameter gradients

        for _input in input_list:
            loss, softmax_scores = self.model(**_input)
            loss.backward()
        
        self.optimizer.step()

        # Clip the norm of the gradients to prevent the "exploding gradients" problem
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def __call__(self, q_features: List[Dict],
                       c_features_list: List[List[Dict]]) -> Union[List[int], List[np.ndarray]]:

        self.model.eval()

        input_list = []
        for i in range(len(q_features)):
            _input = {}
            for elem in ['input_ids', 'attention_mask']:
                inp_elem = [q_features[i][elem]]
                _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
            for elem in ['input_ids', 'attention_mask']:
                inp_elem = [f[elem] for f in c_features_list[i]]
                _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
            input_list.append(_input)

        pred = []
        with torch.no_grad():
            for _input in input_list:
                softmax_scores = self.model(**_input)
                pred_elem = torch.argmax(softmax_scores, dim=1)
                pred_elem = pred_elem.cpu()
                pred_elem = pred_elem.numpy()
                pred.append(list(pred_elem))
        return pred

    def in_batch_accum_ranking_model(self, **kwargs) -> nn.Module:
        return BertAccumRanking(
            pretrained_bert=self.pretrained_bert,
            question_encoder_save_path=self.question_encoder_save_path,
            context_encoder_save_path=self.context_encoder_save_path,
            bert_tokenizer_config_file=self.pretrained_bert,
            device=self.device
        )
        
    def save(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        if fname is None:
            fname = self.save_path
        if not fname.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")
        weights_path = Path(fname).with_suffix(f".pth.tar")
        log.info(f"Saving model to {weights_path}.")
        torch.save({
            "model_state_dict": self.model.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs_done": self.epochs_done
        }, weights_path)
        self.model.to(self.device)
        self.model.save()


class TextEncoder(nn.Module):
    def __init__(self, pretrained_bert: str = None,
                 bert_tokenizer_config_file: str = None,
                 bert_config_file: str = None,
                 device: str = "gpu",
                 bert_emb_size: int = 768,
                 enc_vec_size: int = 300,
                 resize: bool = False
                 ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.device = device
        self.encoder, self.config, self.bert_config = None, None, None
        self.load()
        if Path(bert_tokenizer_config_file).is_file():
            vocab_file = str(expand_path(bert_tokenizer_config_file))
            self.tokenizer = BertTokenizer(vocab_file=vocab_file)
        else:
            tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
        if resize:
            self.encoder.resize_token_embeddings(len(tokenizer) + 1)
        self.bert_emb_size = bert_emb_size
        self.enc_vec_size = enc_vec_size
        self.dense = nn.Linear(self.bert_emb_size, self.enc_vec_size).to(self.device)
        
    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor,
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        hidden_states, cls_emb, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc_vec = self.dense(cls_emb)
        return enc_vec
            
    def load(self) -> None:
        if self.pretrained_bert:
            log.info(f"From pretrained {self.pretrained_bert}.")
            self.config = AutoConfig.from_pretrained(
                self.pretrained_bert, output_hidden_states=True
            )
            self.encoder = BertModel.from_pretrained(self.pretrained_bert, config=self.config)

        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))
            self.encoder = BertModel.from_config(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.encoder.to(self.device)


class BertAccumRanking(nn.Module):

    def __init__(
            self,
            question_encoder_save_path: str,
            context_encoder_save_path: str,
            pretrained_bert: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            c_batch_size: int = 3,
            device: str = "gpu"
    ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.question_encoder_save_path = question_encoder_save_path
        self.context_encoder_save_path = context_encoder_save_path
        self.bert_tokenizer_config_file = bert_tokenizer_config_file
        self.bert_config_file = bert_config_file
        self.c_batch_size = c_batch_size
        self.device = device

        self.q_encoder = TextEncoder(pretrained_bert = self.pretrained_bert,
                                     bert_config_file = self.bert_config_file,
                                     bert_tokenizer_config_file = self.bert_tokenizer_config_file,
                                     device = self.device)
        self.c_encoder = TextEncoder(pretrained_bert = self.pretrained_bert,
                                     bert_config_file = self.bert_config_file,
                                     bert_tokenizer_config_file = self.bert_tokenizer_config_file,
                                     device = self.device)
        self.load()

    def forward(
            self,
            q_input_ids: Tensor,
            q_attention_mask: Tensor,
            c_input_ids: Tensor,
            c_attention_mask: Tensor,
            positive_idx: List[List[int]] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        q_enc_vec = self.q_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask)
        c_enc_vec = self.c_encoder(input_ids=c_input_ids, attention_mask=c_attention_mask)
        if len(q_enc_vec) == 1:
            dot_products = torch.matmul(q_enc_vec, torch.transpose(c_enc_vec, 0, 1))
        else:
            dot_products = torch.matmul(c_enc_vec, torch.transpose(q_enc_vec, 0, 1))
        softmax_scores = F.log_softmax(dot_products, dim=1)
        if positive_idx is not None:
            loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx).to(softmax_scores.device), reduction="mean")
            return loss, softmax_scores
        else:
            return softmax_scores

    def load(self) -> None:
        self.q_encoder.load()
        self.c_encoder.load()
        
    def save(self) -> None:
        question_encoder_weights_path = expand_path(self.question_encoder_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving question encoder to {question_encoder_weights_path}.")
        torch.save({"model_state_dict": self.q_encoder.cpu().state_dict()}, question_encoder_weights_path)
        context_encoder_weights_path = expand_path(self.context_encoder_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving context encoder to {context_encoder_weights_path}.")
        torch.save({"model_state_dict": self.c_encoder.cpu().state_dict()}, context_encoder_weights_path)
        self.q_encoder.to(self.device)
        self.c_encoder.to(self.device)
