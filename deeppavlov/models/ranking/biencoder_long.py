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

log = getLogger(__name__)


@register('torch_transformers_bienc_ranker_long')
class BertBiencRankerLong(TorchModel):

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
            distributed: List[int] = None,
            device: str = "gpu",
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
        self.distributed = distributed
        if self.distributed:
            self.device = f"cuda:{self.distributed[0]}"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")

        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            criterion=criterion,
            optimizer_parameters=optimizer_parameters,
            return_probas=return_probas,
            device=self.device,
            **kwargs)

    def train_on_batch(self, q_features: List[Dict],
                             c_features_list: List[List[Dict]],
                             positive_idx: List[List[int]]) -> float:
        
        _input = {'positive_idx': positive_idx}
        
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            inp_elem = [f[elem] for f in q_features]
            _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            inp_elem = [f[elem] for c_features in c_features_list for f in c_features]
            _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
        
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()      # zero the parameter gradients

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

        _input = {}
        
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            inp_elem = [f[elem] for f in q_features]
            _input[f"q_{elem}"] = torch.LongTensor(inp_elem).to(self.device)
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            inp_elem = [f[elem] for c_features in c_features_list for f in c_features]
            _input[f"c_{elem}"] = torch.LongTensor(inp_elem).to(self.device)

        with torch.no_grad():
            softmax_scores = self.model(**_input)
            pred = torch.argmax(softmax_scores, dim=1)
            pred = pred.cpu()
            pred = pred.numpy()
        return pred

    def in_batch_bienc_long_ranking_model(self, **kwargs) -> nn.Module:
        return BertBiencRankingLong(
            pretrained_bert=self.pretrained_bert,
            question_encoder_save_path=self.question_encoder_save_path,
            context_encoder_save_path=self.context_encoder_save_path,
            bert_tokenizer_config_file=self.pretrained_bert,
            distributed=self.distributed,
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


class TextEncoderLong(nn.Module):
    def __init__(self, pretrained_bert: str = None,
                 bert_tokenizer_config_file: str = None,
                 bert_config_file: str = None,
                 resize: bool = False
                 ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.encoder, self.config, self.bert_config = None, None, None
        self.load()
        if Path(bert_tokenizer_config_file).is_file():
            vocab_file = str(expand_path(bert_tokenizer_config_file))
            self.tokenizer = BertTokenizer(vocab_file=vocab_file)
        else:
            tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
        if resize:
            self.encoder.resize_token_embeddings(len(tokenizer) + 1)
        
    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor,
                token_type_ids: Tensor
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        hidden_states, cls_emb, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return cls_emb
            
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


class BiencoderLong(nn.Module):
    def __init__(
            self,
            question_encoder_save_path: str,
            context_encoder_save_path: str,
            pretrained_bert: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            distributed: List[int] = None
    ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.question_encoder_save_path = question_encoder_save_path
        self.context_encoder_save_path = context_encoder_save_path
        self.bert_tokenizer_config_file = bert_tokenizer_config_file
        self.bert_config_file = bert_config_file

        self.q_encoder = TextEncoderLong(pretrained_bert = self.pretrained_bert,
                                     bert_config_file = self.bert_config_file,
                                     bert_tokenizer_config_file = self.bert_tokenizer_config_file)
        self.c_encoder = TextEncoderLong(pretrained_bert = self.pretrained_bert,
                                     bert_config_file = self.bert_config_file,
                                     bert_tokenizer_config_file = self.bert_tokenizer_config_file)
        self.load()
    
    def forward(
            self,
            q_input_ids: Tensor,
            q_attention_mask: Tensor,
            q_token_type_ids: Tensor,
            c_input_ids: Tensor,
            c_attention_mask: Tensor,
            c_token_type_ids: Tensor,
            positive_idx: List[List[int]] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        q_enc_vec = self.q_encoder(input_ids=q_input_ids, attention_mask=q_attention_mask, token_type_ids=q_token_type_ids)
        c_enc_vec = self.c_encoder(input_ids=c_input_ids, attention_mask=c_attention_mask, token_type_ids=c_token_type_ids)
        
        return q_enc_vec, c_enc_vec

    def load(self) -> None:
        self.q_encoder.load()
        self.c_encoder.load()
        
    def save(self) -> None:
        print("--------------------saving")
        question_encoder_weights_path = expand_path(self.question_encoder_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving question encoder to {question_encoder_weights_path}.")
        torch.save({"model_state_dict": self.q_encoder.state_dict()}, question_encoder_weights_path)
        context_encoder_weights_path = expand_path(self.context_encoder_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving context encoder to {context_encoder_weights_path}.")
        torch.save({"model_state_dict": self.c_encoder.state_dict()}, context_encoder_weights_path)


class BertBiencRankingLong(nn.Module):

    def __init__(
            self,
            question_encoder_save_path: str,
            context_encoder_save_path: str,
            device: str,
            pretrained_bert: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            distributed: List[int] = None
    ):
        super().__init__()
        self.distributed = distributed
        self.device = device
        self.biencoder = BiencoderLong(question_encoder_save_path=question_encoder_save_path,
                                   context_encoder_save_path=context_encoder_save_path,
                                   pretrained_bert=pretrained_bert,
                                   bert_tokenizer_config_file=bert_tokenizer_config_file,
                                   bert_config_file=bert_config_file)
        self.biencoder.load()
        if self.distributed:
            self.biencoder = torch.nn.DataParallel(self.biencoder, device_ids=self.distributed)
        self.biencoder.to(self.device)

    def forward(
            self,
            q_input_ids: Tensor,
            q_attention_mask: Tensor,
            q_token_type_ids: Tensor,
            c_input_ids: Tensor,
            c_attention_mask: Tensor,
            c_token_type_ids: Tensor,
            positive_idx: List[List[int]] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        q_enc_vec, c_enc_vec = self.biencoder(q_input_ids=q_input_ids, q_attention_mask=q_attention_mask, q_token_type_ids=q_token_type_ids,
                                              c_input_ids=c_input_ids, c_attention_mask=c_attention_mask, c_token_type_ids=c_token_type_ids)
        
        dot_products = torch.matmul(q_enc_vec, torch.transpose(c_enc_vec, 0, 1))
        
        softmax_scores = F.log_softmax(dot_products, dim=1)
        if positive_idx is not None:
            loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx).to(self.device), reduction="mean")
            return loss, softmax_scores
        else:
            return softmax_scores
        
    def save(self) -> None:
        if self.distributed:
            self.biencoder.module.save()
        else:
            self.biencoder.save()
