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

        loss, softmax_scores = self.model(**_input)
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
                       text_features: List[Dict]) -> Union[List[int], List[np.ndarray]]:

        self.model.eval()
        _input = {'source_ids': source_ids, 'target_ids': target_ids}
        
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            _input[f"text_{elem}"] = torch.LongTensor(text_features[elem]).to(self.device)

        with torch.no_grad():
            softmax_scores = self.model(**_input)
            pred = torch.argmax(softmax_scores, dim=1)
            pred = pred.cpu()
            pred = pred.numpy()
        return pred

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
        self.token_classifier = nn.Linear(768, 2)

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
        tokens_hidden = hidden_states[:, 1:, :]
        cls_logits = self.classifier(cls_hidden)
        token_logits = self.token_classifier(tokens_hidden)
        logits = torch.cat((cls_logits, token_logits), 1)

        loss = None
        if labels is not None:
            #print("labels list", len(labels), [len(elem) for elem in labels])
            labels = torch.LongTensor(labels).to(self.device)
            #print("logits size", logits.size())
            #print("labels size", labels.size())
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if text_attention_mask is not None:
                active_loss = text_attention_mask.view(-1) == 1
                #print("active loss size", active_loss.size())
                active_logits = logits.view(-1, 2)
                #print("active logits size", active_logits.size())
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                #print("active_labels", active_labels)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
        cls_token_logits = logits[:, :1, :]
        cls_token_logits = torch.squeeze(cls_token_logits, dim=1)
        #print("cls_token_logits", cls_token_logits)
        
        if labels is not None:
            return loss, cls_token_logits
        else:
            return cls_token_logits
        
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
