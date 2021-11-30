import itertools
import pickle
import time
from pathlib import Path
from logging import getLogger
from typing import List, Optional, Dict, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from deeppavlov.core.commands.utils import expand_path
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertTokenizer, BertModel
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


def softmax_mask(val, mask):
    inf = 1e30
    return -inf * (1 - mask.to(torch.float32)) + val


@register('copy_define_model_ind')
class CopyDefineModelInd(TorchModel):

    def __init__(
            self,
            model_name: str,
            encoder_save_path: str,
            emb_save_path: str,
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
            devices: List[int] = None,
            **kwargs
    ):
        self.encoder_save_path = encoder_save_path
        self.emb_save_path = emb_save_path
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.return_probas = return_probas
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm
        self.devices = devices
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert, do_lower_case=True)
        special_tokens_dict = {'additional_special_tokens': ['<text>', '<ner>', '</ner>', '<freq_topic>', '</freq_topic>']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            criterion=criterion,
            optimizer_parameters=optimizer_parameters,
            return_probas=return_probas,
            devices=self.devices,
            **kwargs)

    def train_on_batch(self, source_ids: List[int],
                             target_ids: List[int],
                             text_features: List[Dict],
                             entities_ind_sent: List[List[int]],
                             topic_inds: List[List[int]],
                             token_inds: List[List[int]],
                             cls_labels: List[int],
                             topic_labels: List[List[int]],
                             token_labels: List[List[int]],
                             sentiment: List[List[int]]) -> float:
        
        _input = {'source_ids': source_ids, 'target_ids': target_ids, 'entities_ind_sent': entities_ind_sent,
                  'topic_inds': topic_inds, 'token_inds': token_inds, 'cls_labels': cls_labels,
                  'topic_labels': topic_labels, 'token_labels': token_labels, 'sentiment': sentiment}
        
        if isinstance(text_features, list) or isinstance(text_features, tuple):
            max_len = max([len(elem) for elem in text_features]) + 2
            input_ids_batch = []
            attention_mask_batch = []
            token_type_ids_batch = []
            for wordpiece_tokens in text_features:
                encoding = self.tokenizer.encode_plus(wordpiece_tokens, add_special_tokens = True,
                                                  truncation = True, max_length=max_len,
                                                  pad_to_max_length=True, return_attention_mask = True)
                input_ids_batch.append(encoding["input_ids"][:490])
                attention_mask_batch.append(encoding["attention_mask"][:490])
                token_type_ids_batch.append(encoding["token_type_ids"][:490])
            _input["text_input_ids"] = torch.LongTensor(input_ids_batch).to(self.device)
            _input["text_attention_mask"] = torch.LongTensor(attention_mask_batch).to(self.device)
            _input["text_token_type_ids"] = torch.LongTensor(token_type_ids_batch).to(self.device)
        else:
            for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
                _input[f"text_{elem}"] = torch.LongTensor(text_features[elem]).to(self.device)
        
        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()      # zero the parameter gradients

        loss, *_ = self.model(**_input)
        loss.backward()
        
        self.optimizer.step()

        # Clip the norm of the gradients to prevent the "exploding gradients" problem
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def __call__(self, source_ids: List[int],
                       target_ids: List[int],
                       text_features: List[Dict],
                       entities_ind_sent: List[List[int]],
                       topic_inds: List[List[int]],
                       token_inds: List[List[int]]) -> Union[List[int], List[np.ndarray]]:

        self.model.eval()
        _input = {'source_ids': source_ids, 'target_ids': target_ids, 'entities_ind_sent': entities_ind_sent,
                  'topic_inds': topic_inds, 'token_inds': token_inds}
        
        if isinstance(text_features, list) or isinstance(text_features, tuple):
            max_len = max([len(elem) for elem in text_features]) + 2
            input_ids_batch = []
            attention_mask_batch = []
            token_type_ids_batch = []
            for wordpiece_tokens in text_features:
                encoding = self.tokenizer.encode_plus(wordpiece_tokens, add_special_tokens = True,
                                                  truncation = True, max_length=max_len,
                                                  pad_to_max_length=True, return_attention_mask = True)
                input_ids_batch.append(encoding["input_ids"][:490])
                attention_mask_batch.append(encoding["attention_mask"][:490])
                token_type_ids_batch.append(encoding["token_type_ids"][:490])
            _input["text_input_ids"] = torch.LongTensor(input_ids_batch).to(self.device)
            _input["text_attention_mask"] = torch.LongTensor(attention_mask_batch).to(self.device)
            _input["text_token_type_ids"] = torch.LongTensor(token_type_ids_batch).to(self.device)
        else:
            for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
                _input[f"text_{elem}"] = torch.LongTensor(text_features[elem]).to(self.device)
        copy_topic_inds = []
        copy_token_inds = []
        
        tm_st = time.time()
        with torch.no_grad():
            cls_softmax_scores, topic_softmax_scores, token_softmax_scores, sent_softmax_scores = self.model(**_input)
            copy_or_not = torch.argmax(cls_softmax_scores, dim=1)
            copy_or_not_conf, _ = torch.max(cls_softmax_scores, dim=1)
            copy_or_not = copy_or_not.cpu().numpy()
            copy_or_not_conf = copy_or_not_conf.cpu().numpy()
            
            if (isinstance(topic_softmax_scores, torch.Tensor) and isinstance(token_softmax_scores, torch.Tensor)) or \
                    (topic_softmax_scores and token_softmax_scores):
                topic_preds = torch.argmax(topic_softmax_scores, dim=2).cpu().numpy().tolist()
                token_preds = torch.argmax(token_softmax_scores, dim=2).cpu().numpy().tolist()
                for topic_ind, topic_pred in zip(topic_inds, topic_preds):
                    copy_topic_ind = []
                    for i in range(len(topic_ind)):
                        if topic_pred[i] == 1:
                            copy_topic_ind.append(topic_ind[i] - 1)
                    copy_topic_inds.append(copy_topic_ind)
                
                for token_ind, token_pred in zip(token_inds, token_preds):
                    copy_token_ind = []
                    for i in range(len(token_ind)):
                        if token_pred[i] == 1:
                            copy_token_ind.append(token_ind[i] - 19)
                    copy_token_inds.append(copy_token_ind)
            else:
                copy_topic_inds = [[] for _ in source_ids]
                copy_token_inds = [[] for _ in source_ids]
            
        if isinstance(sent_softmax_scores, list) and not sent_softmax_scores:
            sent_softmax_scores = [[] for _ in source_ids]
        else:
            sent_softmax_scores = F.softmax(sent_softmax_scores, dim=2)
            sent_softmax_scores = torch.argmax(sent_softmax_scores, dim=2)
            sent_softmax_scores = sent_softmax_scores.cpu().numpy().tolist()
            
        return copy_or_not, copy_or_not_conf, copy_topic_inds, copy_token_inds, sent_softmax_scores

    def copy_define_model(self, **kwargs) -> nn.Module:
        return CopyDefineNetwork(
            pretrained_bert=self.pretrained_bert,
            encoder_save_path=self.encoder_save_path,
            emb_save_path=self.emb_save_path,
            bert_tokenizer_config_file=self.pretrained_bert,
            devices=self.devices
        )
        
    def save(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        self.model.save()
        
    def load(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        model_func = getattr(self, self.opt.get("model_name"), None)
        self.init_from_opt(model_func)
        self.model.load()


class TextEncoder(nn.Module):
    def __init__(self, pretrained_bert: str = None,
                 bert_tokenizer_config_file: str = None,
                 bert_config_file: str = None,
                 device: str = "gpu"
                 ):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.encoder, self.config, self.bert_config = None, None, None
        self.device = device
        self.load()
        tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
        self.encoder.resize_token_embeddings(len(tokenizer) + 5)
        
    def forward(self,
                input_ids: Tensor,
                attention_mask: Tensor,
                token_type_ids: Tensor
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        c_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return c_outputs
            
    def load(self) -> None:
        if self.pretrained_bert:
            log.info(f"From pretrained {self.pretrained_bert}.")
            self.config = AutoConfig.from_pretrained(
                self.pretrained_bert, output_hidden_states=True
            )
            self.encoder = AutoModel.from_pretrained(self.pretrained_bert, config=self.config)

        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.config = AutoConfig.from_json_file(str(expand_path(self.bert_config_file)))
            self.encoder = AutoModel.from_config(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")


class CopyDefineNetwork(nn.Module):

    def __init__(
            self,
            encoder_save_path: str,
            emb_save_path: str,
            pretrained_bert: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            device: str = "gpu",
            devices: List[int] = None  
    ):
        super().__init__()
        self.devices = devices
        if self.devices is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        else:
            self.device = torch.device(f"cuda:{self.devices[0]}")
        self.pretrained_bert = pretrained_bert
        self.encoder_save_path = encoder_save_path
        self.emb_save_path = emb_save_path
        self.bert_config_file = bert_config_file
        self.encoder, self.config, self.bert_config = None, None, None
        self.zero_vec = torch.Tensor(768)
        self.source_embeddings = nn.Embedding(862, 384)
        self.target_embeddings = nn.Embedding(862, 384)
        self.bilinear_cls = nn.Linear(768 * 8, 2)
        self.bilinear_topic = nn.Linear(768 * 8, 2)
        self.bilinear_token = nn.Linear(768 * 8, 2)
        self.bilinear_sent = nn.Linear(768 * 8, 5)
        self.encoder = TextEncoder(pretrained_bert=self.pretrained_bert,
                                   bert_tokenizer_config_file=bert_tokenizer_config_file,
                                   bert_config_file=bert_config_file, device=self.device)
        if self.devices is not None:
            print("----------------- multi gpu, in load")
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=self.devices)
        self.tm_enc = 0.0
        self.tm_bilinear = 0.0

    def forward(
            self,
            source_ids: List[int],
            target_ids: List[int],
            text_input_ids: Tensor,
            text_attention_mask: Tensor,
            text_token_type_ids: Tensor,
            entities_ind_sent: List[List[int]],
            topic_inds: List[List[int]],
            token_inds: List[List[int]],
            cls_labels: List[int] = None,
            topic_labels: List[List[int]] = None,
            token_labels: List[List[int]] = None,
            sentiment: List[int] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        bs = text_attention_mask.shape[0]
        
        if topic_labels is not None:
            topic_labels = list(topic_labels)
        if token_labels is not None:
            token_labels = list(token_labels)
        source_ids = torch.LongTensor(source_ids).to(self.device)
        target_ids = torch.LongTensor(target_ids).to(self.device)
        source_embs = self.source_embeddings(source_ids)
        target_embs = self.target_embeddings(target_ids)
        domain_embs = torch.cat((source_embs, target_embs), 1)
        domain_embs = torch.unsqueeze(domain_embs, 1).to(self.device)
        self.tm_enc = 0.0
        self.tm_bilinear = 0.0
        tm1 = time.time()
        bert_output = self.encoder(input_ids=text_input_ids,
                                   attention_mask=text_attention_mask,
                                   token_type_ids=text_token_type_ids)
        hidden_states = bert_output.last_hidden_state
        self.tm_enc = time.time() - tm1
        
        tm1 = time.time()
        entities_sent_batch = []
        for i in range(bs):
            entities_sent_list = []
            for j in range(len(entities_ind_sent[i])):
                ind = entities_ind_sent[i][j] + 1
                entities_sent_list.append(hidden_states[i][ind])
            entities_sent_batch.append(entities_sent_list)
        
        sent_att_mask_batch = []
        
        max_entities_sent_len = max([len(elem) for elem in entities_sent_batch])
        if sentiment is not None:
            sentiment = list(sentiment)
        
        if max_entities_sent_len > 0:
            for i in range(bs):
                sent_att_mask = [0 for _ in range(max_entities_sent_len)]
                for j in range(len(entities_sent_batch[i])):
                    sent_att_mask[j] = 1
                for j in range(max_entities_sent_len - len(entities_sent_batch[i])):
                    entities_sent_batch[i].append(self.zero_vec.to(self.device))
                    if sentiment is not None:
                        sentiment[i].append(0)
                if sentiment is not None:
                    sentiment[i] = sentiment[i][:max_entities_sent_len]
                entities_sent_batch[i] = torch.stack(entities_sent_batch[i], dim=0)
                sent_att_mask_batch.append(sent_att_mask)
                    
            entities_sent_batch = torch.stack(entities_sent_batch, dim=0).to(self.device)
        
        # _______________________________________________________________________________
        
        topics_batch = []
        for i in range(bs):
            topics_list = []
            for j in range(len(topic_inds[i])):
                ind = topic_inds[i][j]
                topics_list.append(hidden_states[i][ind])
            topics_batch.append(topics_list)
        
        topic_att_mask_batch = []
        max_topic_len = max([len(elem) for elem in topics_batch])
        
        if max_topic_len > 0:
            for i in range(bs):
                topic_att_mask = [0 for _ in range(max_topic_len)]
                for j in range(len(topics_batch[i])):
                    topic_att_mask[j] = 1
                for j in range(max_topic_len - len(topics_batch[i])):
                    topics_batch[i].append(self.zero_vec.to(self.device))
                if topic_labels is not None:
                    if max_topic_len > len(topic_labels[i]):
                        for j in range(max_topic_len - len(topic_labels[i])):
                            topic_labels[i].append(0)
                    elif max_topic_len < len(topic_labels[i]):
                        topic_labels[i] = topic_labels[i][:max_topic_len]
                
                topics_batch[i] = torch.stack(topics_batch[i], dim=0)
                topic_att_mask_batch.append(topic_att_mask)
                    
            topics_batch = torch.stack(topics_batch, dim=0).to(self.device)
        
        # _______________________________________________________________________________
        
        tokens_batch = []
        for i in range(bs):
            tokens_list = []
            for j in range(len(token_inds[i])):
                ind = token_inds[i][j]
                tokens_list.append(hidden_states[i][ind])
            tokens_batch.append(tokens_list)
        
        token_att_mask_batch = []
        max_token_len = max([len(elem) for elem in tokens_batch])
        
        if max_token_len > 0:
            for i in range(bs):
                token_att_mask = [0 for _ in range(max_token_len)]
                for j in range(len(tokens_batch[i])):
                    token_att_mask[j] = 1
                for j in range(max_token_len - len(tokens_batch[i])):
                    tokens_batch[i].append(self.zero_vec.to(self.device))
                if token_labels is not None:
                    if max_token_len > len(token_labels[i]):
                        for j in range(max_token_len - len(token_labels[i])):
                            token_labels[i].append(0)
                    elif max_token_len < len(token_labels[i]):
                        token_labels[i] = token_labels[i][:max_token_len]
                
                tokens_batch[i] = torch.stack(tokens_batch[i], dim=0)
                token_att_mask_batch.append(token_att_mask)
            tokens_batch = torch.stack(tokens_batch, dim=0).to(self.device)
        
        cls_hidden = hidden_states[:, :1, :]
        
        domain_embs0 = domain_embs.view(-1, 96, 8)
        domain_embs1 = domain_embs0.unsqueeze(1)
        
        cls_hidden = torch.squeeze(cls_hidden, 1)
        cls_hidden = cls_hidden.view(-1, 96, 8)
        if max_entities_sent_len > 0:
            entities_sent_batch = entities_sent_batch.view(bs, -1, 96, 8)
        
        bl_cls = (domain_embs0.unsqueeze(3) * cls_hidden.unsqueeze(2)).view(-1, 768 * 8)
        cls_logits = self.bilinear_cls(bl_cls)
        cls_logits = torch.squeeze(cls_logits, dim=1)
        cls_logits = F.softmax(cls_logits, 1)
        
        topic_logits = []
        if (isinstance(topics_batch, list) and topics_batch and topics_batch[0]) or isinstance(topics_batch, torch.Tensor):
            topic_hidden = topics_batch.view(bs, -1, 96, 8)
            bl_topic = (domain_embs1.unsqueeze(4) * topic_hidden.unsqueeze(3)).view(bs, -1, 768 * 8)
            topic_logits = self.bilinear_topic(bl_topic)
            topic_logits = F.softmax(topic_logits, 2)
        
        token_logits = []
        if (isinstance(tokens_batch, list) and tokens_batch and tokens_batch[0]) or isinstance(tokens_batch, torch.Tensor):
            token_hidden = tokens_batch.view(bs, -1, 96, 8)
            bl_token = (domain_embs1.unsqueeze(4) * token_hidden.unsqueeze(3)).view(bs, -1, 768 * 8)
            token_logits = self.bilinear_token(bl_token)
            token_logits = F.softmax(token_logits, 2)
        
        if max_entities_sent_len > 0:
            bl_sent = (domain_embs1.unsqueeze(4) * entities_sent_batch.unsqueeze(3)).view(bs, -1, 768 * 8)
            sent_logits = self.bilinear_sent(bl_sent)
        else:
            sent_logits = []
        
        topic_att_mask_batch = torch.LongTensor(topic_att_mask_batch).to(self.device)
        token_att_mask_batch = torch.LongTensor(token_att_mask_batch).to(self.device)
        sent_att_mask_batch = torch.LongTensor(sent_att_mask_batch).to(self.device)
        
        ce_loss_fct = nn.CrossEntropyLoss()
        self.tm_bilinear = time.time() - tm1
        
        total_loss = None
        if sentiment is not None:
            sentiment = torch.LongTensor(sentiment).to(self.device) 
            topic_labels = torch.LongTensor(topic_labels).to(self.device)
            token_labels = torch.LongTensor(token_labels).to(self.device)
            
            cls_labels = torch.LongTensor(cls_labels).to(self.device)
            cls_loss = ce_loss_fct(cls_logits, cls_labels)
            
            total_loss = 0.0
            if all([elem == 1 for elem in cls_labels]):
                if not isinstance(sent_logits, list):
                    active_loss = sent_att_mask_batch.view(-1) == 1
                    active_logits = sent_logits.view(-1, 5)
                    active_labels = torch.where(
                        active_loss, sentiment.view(-1), torch.tensor(ce_loss_fct.ignore_index).type_as(sentiment)
                    )
                    sent_loss = ce_loss_fct(active_logits, active_labels)
                    total_loss += sent_loss
                
                if not isinstance(topic_logits, list):
                    active_loss = topic_att_mask_batch.view(-1) == 1
                    active_logits = topic_logits.view(-1, 2)
                    active_labels = torch.where(
                        active_loss, topic_labels.view(-1), torch.tensor(ce_loss_fct.ignore_index).type_as(topic_labels)
                    )
                    topic_loss = ce_loss_fct(active_logits, active_labels)
                    total_loss += topic_loss
                    
                if not isinstance(token_logits, list):
                    active_loss = token_att_mask_batch.view(-1) == 1
                    active_logits = token_logits.view(-1, 2)
                    active_labels = torch.where(
                        active_loss, token_labels.view(-1), torch.tensor(ce_loss_fct.ignore_index).type_as(token_labels)
                    )
                    token_loss = ce_loss_fct(active_logits, active_labels)
                    total_loss += token_loss
            else:
                total_loss = cls_loss

        if sentiment is not None:
            return total_loss, cls_logits, topic_logits, token_logits, sent_logits
        else:
            return cls_logits, topic_logits, token_logits, sent_logits
        
    def save(self) -> None:
        print("--------------------saving")
        encoder_weights_path = expand_path(self.encoder_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving encoder to {encoder_weights_path}.")
        if self.devices is not None:
            encoder_state_dict = self.encoder.module.encoder.state_dict()
        else:
            encoder_state_dict = self.encoder.encoder.state_dict()
        torch.save({"encoder_state_dict": encoder_state_dict,
                    "cls_state_dict": self.bilinear_cls.state_dict(),
                    "topic_state_dict": self.bilinear_topic.state_dict(),
                    "token_state_dict": self.bilinear_token.state_dict(),
                    "sent_state_dict": self.bilinear_sent.state_dict()}, encoder_weights_path)
        emb_weights_path = str(expand_path(self.emb_save_path))
        indices = [i for i in range(862)]
        indices = torch.LongTensor(indices).to(self.device)
        source_vectors = self.source_embeddings(indices)
        target_vectors = self.target_embeddings(indices)
        source_vectors = source_vectors.detach().cpu().numpy().tolist()
        target_vectors = target_vectors.detach().cpu().numpy().tolist()
        with open(f"{emb_weights_path}/source_domain_vectors.pickle", 'wb') as out:
            pickle.dump(source_vectors, out)
        with open(f"{emb_weights_path}/target_domain_vectors.pickle", 'wb') as out:
            pickle.dump(target_vectors, out)
    
    def load(self) -> None:
        encoder_weights_path = expand_path(self.encoder_save_path).with_suffix(f".pth.tar")
        if Path(encoder_weights_path).exists():
            log.info(f"--------------- Loading encoder to {encoder_weights_path}.")
            checkpoint = torch.load(encoder_weights_path, map_location=self.device)
            if self.devices is None:
                self.encoder.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            else:
                self.encoder.module.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.bilinear_cls.load_state_dict(checkpoint["cls_state_dict"])
            self.bilinear_topic.load_state_dict(checkpoint["topic_state_dict"])
            self.bilinear_token.load_state_dict(checkpoint["token_state_dict"])
            self.bilinear_sent.load_state_dict(checkpoint["sent_state_dict"])
            
        self.encoder.to(self.device)
        self.bilinear_cls.to(self.device)
        self.bilinear_topic.to(self.device)
        self.bilinear_token.to(self.device)
        self.bilinear_sent.to(self.device)
        
        emb_weights_path = str(expand_path(self.emb_save_path))
        init_emb_weights_path = f"{emb_weights_path}/init_domain_vectors.pickle"
        source_emb_weights_path = f"{emb_weights_path}/source_domain_vectors.pickle"
        target_emb_weights_path = f"{emb_weights_path}/target_domain_vectors.pickle"
        if Path(source_emb_weights_path).exists() and Path(target_emb_weights_path).exists():
            log.info("loaded trained embeddings")
            with open(source_emb_weights_path, 'rb') as fl:
                source_emb_weights = pickle.load(fl)
            source_emb_weights = torch.FloatTensor(source_emb_weights)
            self.source_embeddings = nn.Embedding.from_pretrained(source_emb_weights)
            with open(target_emb_weights_path, 'rb') as fl:
                target_emb_weights = pickle.load(fl)
            target_emb_weights = torch.FloatTensor(target_emb_weights)
            self.target_embeddings = nn.Embedding.from_pretrained(target_emb_weights)
        elif Path(init_emb_weights_path).exists():
            log.info("loaded init domain embeddings")
            with open(init_emb_weights_path, 'rb') as fl:
                init_emb_weights = pickle.load(fl)
            init_emb_weights = torch.FloatTensor(init_emb_weights)
            self.source_embeddings = nn.Embedding.from_pretrained(init_emb_weights)
            self.target_embeddings = nn.Embedding.from_pretrained(init_emb_weights)
        
        self.source_embeddings.to(self.device)
        self.target_embeddings.to(self.device)
