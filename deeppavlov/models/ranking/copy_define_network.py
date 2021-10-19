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
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertTokenizer
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import load_pickle
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.ranking.modeling_bert import BertModel

log = getLogger(__name__)


def softmax_mask(val, mask):
    inf = 1e30
    return -inf * (1 - mask.to(torch.float32)) + val


@register('copy_define_model')
class CopyDefineModel(TorchModel):

    def __init__(
            self,
            model_name: str,
            encoder_save_path: str,
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
            **kwargs
    ):
        self.encoder_save_path = encoder_save_path
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

    def train_on_batch(self, source_ids: List[int],
                             target_ids: List[int],
                             text_features: List[Dict],
                             labels: List[List[int]]) -> float:
        
        _input = {'source_ids': source_ids, 'target_ids': target_ids, 'labels': labels}
        
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
    
    def vertical_scores(self, cls_softmax_scores, topic_softmax_scores, token_softmax_scores):
        pred = torch.argmax(cls_softmax_scores, dim=1)
        topic_pred = torch.argmax(topic_softmax_scores, dim=2)
        token_pred = torch.argmax(token_softmax_scores, dim=2)
        pred = pred.cpu()
        pred = pred.numpy()
        topic_pred = topic_pred.cpu()
        topic_pred_batch = topic_pred.numpy().tolist()
        token_pred = token_pred.cpu()
        token_pred_batch = token_pred.numpy().tolist()
        
        topic_ind_batch, token_ind_batch = [], []
        for topic_pred, token_pred in zip(topic_pred_batch, token_pred_batch):
            topic_ind_list, token_ind_list = [], []
            for i in range(len(topic_pred)):
                if topic_pred[i] == 1:
                    topic_ind_list.append(i)
            for i in range(len(token_pred)):
                if token_pred[i] == 1:
                    token_ind_list.append(i)
            topic_ind_batch.append(topic_ind_list)
            token_ind_batch.append(token_ind_list)
        
        return pred, topic_ind_batch, token_ind_batch
    
    def horizontal_scores(self, cls_softmax_scores, topic_softmax_scores, token_softmax_scores):
        pred = torch.argmax(cls_softmax_scores, dim=1)
        pred = pred.cpu()
        pred = pred.numpy()
        topic_pred = topic_softmax_scores.cpu()
        topic_pred_batch = topic_pred.numpy().tolist()
        token_pred = token_softmax_scores.cpu()
        token_pred_batch = token_pred.numpy().tolist()
        
        av_max_min = []
        thres_batch = []
        for token_pred in token_pred_batch:
            elem_list = []
            for elem in token_pred:
                if elem > 0.0:
                    elem_list.append(elem)
            average = round(sum(elem_list) / len(elem_list), 4)
            maximum = round(max(elem_list), 4)
            minimum = round(min(elem_list), 4)
            av_max_min.append((average, maximum, minimum))
            thres = (maximum + average) / 2
            thres_batch.append(thres)
            
        out = open("log_scores.txt", 'a')
        out.write(str([round(elem, 3) for elem in token_pred_batch[0]])+'\n')
        out.write(str(av_max_min[0])+'\n\n')
        out.close()
        
        topic_ind_batch, token_ind_batch = [], []
        for topic_pred, token_pred, thres in zip(topic_pred_batch, token_pred_batch, thres_batch):
            topic_ind_list, token_ind_list = [], []
            for i in range(len(topic_pred)):
                if topic_pred[i] > 0.005:
                    topic_ind_list.append(i)
            for i in range(len(token_pred)):
                if token_pred[i] > thres:
                    token_ind_list.append(i)
            topic_ind_batch.append(topic_ind_list)
            token_ind_batch.append(token_ind_list)
        
        return pred, topic_ind_batch, token_ind_batch

    def __call__(self, source_ids: List[int],
                       target_ids: List[int],
                       text_features: List[Dict]) -> Union[List[int], List[np.ndarray]]:

        self.model.eval()
        _input = {'source_ids': source_ids, 'target_ids': target_ids}
        
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            _input[f"text_{elem}"] = torch.LongTensor(text_features[elem]).to(self.device)

        with torch.no_grad():
            cls_softmax_scores, topic_softmax_scores, token_softmax_scores = self.model(**_input)
        
        pred, topic_ind_batch, token_ind_batch = \
            self.horizontal_scores(cls_softmax_scores, topic_softmax_scores, token_softmax_scores)
            
        return pred, topic_ind_batch, token_ind_batch

    def copy_define_model(self, **kwargs) -> nn.Module:
        return CopyDefineNetwork(
            pretrained_bert=self.pretrained_bert,
            encoder_save_path=self.encoder_save_path,
            bert_tokenizer_config_file=self.pretrained_bert
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


class CopyDefineNetwork(nn.Module):

    def __init__(
            self,
            encoder_save_path: str,
            pretrained_bert: str = None,
            bert_tokenizer_config_file: str = None,
            bert_config_file: str = None,
            device: str = "gpu"
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.pretrained_bert = pretrained_bert
        self.encoder_save_path = encoder_save_path
        self.bert_config_file = bert_config_file
        self.encoder, self.config, self.bert_config = None, None, None
        self.load()
        if Path(bert_tokenizer_config_file).is_file():
            vocab_file = str(expand_path(bert_tokenizer_config_file))
            self.tokenizer = BertTokenizer(vocab_file=vocab_file)
        else:
            tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
        self.encoder.resize_token_embeddings(len(tokenizer) + 4)
        
        self.classifier = nn.Linear(768, 2)
        self.topic_token_classifier = nn.Linear(768, 1)
        self.token_classifier = nn.Linear(768, 1)

    def forward(
            self,
            source_ids: List[int],
            target_ids: List[int],
            text_input_ids: Tensor,
            text_attention_mask: Tensor,
            text_token_type_ids: Tensor,
            labels: List[int] = None
    ) -> Union[Tuple[Any, Tensor], Tuple[Tensor]]:

        bs = text_attention_mask.shape[0]
        one_tensor = torch.Tensor([[1] for _ in range(bs)]).to(self.device)
        ext_text_attention_mask = torch.cat((one_tensor, text_attention_mask), 1)
        source_ids = torch.LongTensor(source_ids).to(self.device)
        target_ids = torch.LongTensor(target_ids).to(self.device)
        hidden_states, *_ = self.encoder(input_ids=text_input_ids,
                                         attention_mask=ext_text_attention_mask,
                                         token_type_ids=text_token_type_ids,
                                         source_ids=source_ids,
                                         target_ids=target_ids)
        
        cls_hidden = hidden_states[:, :1, :]
        topic_tokens_hidden = hidden_states[:, 1:19, :]
        tokens_hidden = hidden_states[:, 19:, :]
        
        cls_logits = self.classifier(cls_hidden)
        topic_token_logits = self.topic_token_classifier(topic_tokens_hidden)
        token_logits = self.token_classifier(tokens_hidden)
        
        loss, cls_token_logits, topic_token_logits, token_logits = \
            self.get_logits_horizontal_one(text_attention_mask, cls_logits, topic_token_logits, token_logits, labels)
        
        if labels is not None:
            return loss, cls_token_logits, topic_token_logits, token_logits
        else:
            return cls_token_logits, topic_token_logits, token_logits
    
    def get_logits_horizontal_one(self, text_attention_mask, cls_logits, topic_logits, token_logits, labels):
        cls_logits = torch.squeeze(cls_logits, dim=1)
        topic_logits = topic_logits.squeeze(-1).contiguous()
        topic_logits = F.softmax(topic_logits, 1)
        
        token_logits = token_logits.squeeze(-1).contiguous()
        token_attention_mask = text_attention_mask[:, 19:]
        token_logits = softmax_mask(token_logits, token_attention_mask)
        token_logits = F.softmax(token_logits, 1)
        
        ce_loss_fct = nn.CrossEntropyLoss()
        bce_loss_fct = nn.BCEWithLogitsLoss()
        
        total_loss = None
        if labels is not None:
            cls_labels = torch.LongTensor([elem[0] for elem in labels]).to(self.device)
            cls_loss = ce_loss_fct(cls_logits, cls_labels)
            
            if all([elem == 1 for elem in cls_labels]):
                topic_labels = torch.Tensor([elem[1:19] for elem in labels]).to(self.device)
                topic_loss = bce_loss_fct(topic_logits, topic_labels)
                
                token_labels = [elem[19:] for elem in labels]
                token_labels = torch.Tensor(token_labels).to(self.device)
                token_loss = bce_loss_fct(token_logits, token_labels)
                
                total_loss = topic_loss + token_loss
            else:
                total_loss = cls_loss

        return total_loss, cls_logits, topic_logits, token_logits
            
    def get_logits_horizontal(self, cls_logits, topic_logits, token_logits, labels):
        cls_logits = torch.squeeze(cls_logits, dim=1)
        ce_loss_fct = nn.CrossEntropyLoss()
        bce_loss_fct = nn.BCEWithLogitsLoss()
        
        topic_logits = F.softmax(topic_logits, dim=2)
        topic_start_logits, topic_end_logits = topic_logits.split(1, dim=-1)
        topic_start_logits = topic_start_logits.squeeze(-1).contiguous()
        
        token_logits = F.softmax(token_logits, dim=2)
        token_start_logits, token_end_logits = token_logits.split(1, dim=-1)
        token_start_logits = token_start_logits.squeeze(-1).contiguous()
        
        total_loss = None
        if labels is not None:
            cls_labels = torch.LongTensor([elem[0] for elem in labels]).to(self.device)
            cls_loss = ce_loss_fct(cls_logits, cls_labels)
            
            topic_labels = torch.Tensor([elem[1:19] for elem in labels]).to(self.device)
            topic_loss = bce_loss_fct(topic_start_logits, topic_labels)
            
            token_labels = torch.Tensor([elem[19:] for elem in labels]).to(self.device)
            token_loss = bce_loss_fct(token_start_logits, token_labels)
            
            total_loss = cls_loss + topic_loss + token_loss

        return total_loss, cls_logits, topic_start_logits, token_start_logits
    
    def get_logits_vertical(self, cls_logits, topic_logits, token_logits, labels):
        logits = torch.cat((cls_logits, topic_logits), 1)
        logits = torch.cat((logits, token_logits), 1)

        loss = None
        if labels is not None:
            labels = torch.LongTensor(labels).to(self.device)
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if text_attention_mask is not None:
                active_loss = text_attention_mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
        cls_token_logits = logits[:, :1, :]
        cls_token_logits = torch.squeeze(cls_token_logits, dim=1)
        topic_token_logits = logits[:, 1:19, :]
        token_logits = logits[:, 19:, :]
        
        return loss, cls_token_logits, topic_logits, token_logits
        
    def save(self) -> None:
        print("--------------------saving")
        encoder_weights_path = expand_path(self.encoder_save_path).with_suffix(f".pth.tar")
        log.info(f"Saving encoder to {encoder_weights_path}.")
        torch.save({"model_state_dict": self.encoder.state_dict()}, encoder_weights_path)
        
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
