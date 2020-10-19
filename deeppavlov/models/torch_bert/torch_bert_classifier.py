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

from logging import getLogger
from pathlib import Path
from typing import List, Dict, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from overrides import overrides
from transformers import BertForSequenceClassification, BertConfig, BertModel
from transformers.modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


@register('torch_bert_classifier')
class TorchBertClassifierModel(TorchModel):
    """Bert-based model for text classification on PyTorch.

    It uses output from [CLS] token and predicts labels using linear transformation.

    Args:
        n_classes: number of classes
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        one_hot_labels: set True if one-hot encoding for labels is used
        multilabel: set True if it is multi-label classification
        return_probas: set True if return class probabilites instead of most probable label needed
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        optimizer: optimizer name from `torch.optim`
        optimizer_parameters: dictionary with optimizer's parameters,
                              e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
        clip_norm: clip gradients by norm coefficient
        bert_config_file: path to Bert configuration file (not used if pretrained_bert is key title)
    """

    def __init__(self, n_classes,
                 pretrained_bert,
                 one_hot_labels: bool = False,
                 multilabel: bool = False,
                 return_probas: bool = False,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: dict = {"lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.999), "eps": 1e-6},
                 clip_norm: Optional[float] = None,
                 bert_config_file: Optional[str] = None,
                 vocab_size: Optional[int] = None,
                 output_attentions: Optional[bool] = False,
                 output_hidden_states: Optional[bool] = False,
                 pool_mem_tokens: Optional[bool] = False,
                 mem_size: Optional[int] = 0,
                 only_head: Optional[bool] = False,
                 random_init: Optional[bool] = False,
                 mean_max_pool: Optional[bool] = False,
                 **kwargs) -> None:

        self.return_probas = return_probas
        self.one_hot_labels = one_hot_labels
        self.multilabel = multilabel
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        # todo: if n_classes == 1 solve regression task (HF sequence classification models have such logic)
        # todo: add normalization preprocessor and postprocessor
        self.n_classes = n_classes
        self.clip_norm = clip_norm
        self.vocab_size = vocab_size
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        # expr args:
        self.pool_mem_tokens = pool_mem_tokens
        self.mem_size = mem_size
        self.only_head = only_head
        self.random_init = random_init
        self.mean_max_pool = mean_max_pool

        if self.multilabel and not self.one_hot_labels:
            raise RuntimeError('Use one-hot encoded labels for multilabel classification!')

        if self.multilabel and not self.return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

        if self.return_probas and self.n_classes == 1:
            raise RuntimeError('Set return_probas to False for regression task!')

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         **kwargs)

    def train_on_batch(self, features: List[InputFeatures], y: Union[List[int], List[List[int]]]) -> Dict:
        """Train model on given batch.
        This method calls train_op using features and y (labels).

        Args:
            features: batch of InputFeatures
            y: batch of labels (class id or one-hot encoding)

        Returns:
            dict with loss and learning_rate values
        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.attention_mask for f in features]

        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)
        if self.n_classes > 1:
            b_labels = torch.from_numpy(np.array(y)).to(self.device)
        else:
            b_labels = torch.from_numpy(np.array(y, dtype=np.float32)).to(self.device)

        self.optimizer.zero_grad()

        # todo: fix token_type_ids usage
        outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks,
                             labels=b_labels)
        loss = outputs[0]
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    def __call__(self, features: List[InputFeatures]) -> Union[List[int], List[List[float]]]:
        """Make prediction for given features (texts).

        Args:
            features: batch of InputFeatures

        Returns:
            predicted classes or probabilities of each class

        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.attention_mask for f in features]

        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            # todo: fix token_type_ids usage
            output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks)
            logits = output[0]

        if self.return_probas:
            if not self.multilabel:
                pred = torch.nn.functional.softmax(logits, dim=-1)
            else:
                pred = torch.nn.functional.sigmoid(logits)
            pred = pred.detach().cpu().numpy()
        elif self.n_classes > 1:
            logits = logits.detach().cpu().numpy()
            pred = np.argmax(logits, axis=1)
        else:  # regression
            pred = logits.squeeze(-1).detach().cpu().numpy()

        if self.output_attentions or self.output_hidden_states:
            return pred, output[1:]
        return pred

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert and not Path(self.pretrained_bert).is_file():
            self.model = BertForSequenceClassification.from_pretrained(
                self.pretrained_bert, num_labels=self.n_classes,
                output_attentions=self.output_attentions, output_hidden_states=self.output_hidden_states)
        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.bert_config = BertConfig.from_json_file(str(expand_path(self.bert_config_file)))

            if self.attention_probs_keep_prob is not None:
                self.bert_config.attention_probs_dropout_prob = 1.0 - self.attention_probs_keep_prob
            if self.hidden_keep_prob is not None:
                self.bert_config.hidden_dropout_prob = 1.0 - self.hidden_keep_prob
            self.model = BertForSequenceClassification(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        if self.random_init:
            self.model.init_weights()

        if self.pool_mem_tokens and self.mem_size != 0:
            # modify pooling strategy
            class BertPooler(torch.nn.Module):
                def __init__(self, hidden_size=768, pool_start=0, pool_end=1):
                    super().__init__()
                    self.pool_start = pool_start
                    self.pool_end = pool_end
                    self.dense = torch.nn.Linear(hidden_size * 2, hidden_size)
                    self.activation = torch.nn.Tanh()

                def forward(self, hidden_states, attention_mask=None):
                    to_pool = hidden_states[:, self.pool_start:self.pool_end]
                    max_pooled, _ = torch.max(to_pool, dim=1)
                    mean_pooled = torch.mean(to_pool, dim=1)
                    pooled = torch.cat([max_pooled, mean_pooled], dim=-1)
                    pooled_output = self.dense(pooled)
                    pooled_output = self.activation(pooled_output)
                    return pooled_output

            self.model.bert.pooler = BertPooler(hidden_size=self.model.config.hidden_size,
                                                pool_start=1, pool_end=1+self.mem_size)

        if self.mean_max_pool:
            class BertPooler(torch.nn.Module):
                def __init__(self, hidden_size=768):
                    super().__init__()
                    self.INF = 1e30
                    self.dense = torch.nn.Linear(hidden_size * 2, hidden_size)
                    self.activation = torch.nn.Tanh()

                def forward(self, hidden_states, attention_mask=None):
                    mask = attention_mask.unsqueeze(dim=-1)
                    mean_pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
                    max_pooled, _ = (hidden_states - (1 - mask) * self.INF).max(dim=1)
                    pooled = torch.cat([max_pooled, mean_pooled], dim=-1)
                    pooled_output = self.dense(pooled)
                    pooled_output = self.activation(pooled_output)
                    return pooled_output

            self.model.bert.pooler = BertPooler(hidden_size=self.model.config.hidden_size)

        self.model.to(self.device)

        if self.vocab_size:
            vocab_size = self.model.get_input_embeddings().weight.shape[0]
            self.model.resize_token_embeddings(self.vocab_size)
            new_vocab_size = self.model.get_input_embeddings().weight.shape[0]
            if vocab_size != new_vocab_size:
                log.info(f"Embeddings matrix size was changed from {vocab_size} to {new_vocab_size}")

        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        if self.load_path:
            log.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                log.info(f"Load path {weights_path} exists.")
                log.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                log.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.info(f"Init from scratch. Load path {weights_path} does not exist.")

        if self.only_head:
            to_train = ['classifier.weight', 'classifier.bias']
            if self.pool_mem_tokens or self.mean_max_pool:
                to_train += [n for n, p in self.model.named_parameters() if 'bert.pooler' in n]
            for name, p in self.model.named_parameters():
                if name in to_train:
                    p.requires_grad = True
                else:
                    p.requires_grad = False


class MemTower(nn.Module):
    def __init__(self, config, num_labels, mem_size=5):
        super().__init__()
        self.config = config
        self.mem_size = mem_size
        self.num_labels = num_labels
        self.mem_embeddings = nn.Embedding(mem_size, config.hidden_size)
        self.mem_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mem_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tower_layer = TowerLayer(config)
        self.linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.activation = torch.nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.mem_ind = torch.arange(0, self.mem_size).view(1, self.mem_size)

    def forward(self, encoder_hiddens, encoder_attention_mask, labels=None):
        bs = encoder_hiddens[0].shape[0]
        mem = self.mem_embeddings(self.mem_ind.repeat(bs, 1))
        mem = self.mem_layer_norm(mem)
        mem = self.mem_dropout(mem)
        h = mem
        all_hidden_states = (mem, )
        all_attentions = ()
        for i in range(len(encoder_hiddens)):
            layer_outputs = self.tower_layer(h, encoder_hiddens[i], encoder_attention_mask)
            h = layer_outputs[0]
            all_hidden_states = all_hidden_states + (h, )
            if self.config.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        max_pooled, _ = torch.max(h, dim=1)
        mean_pooled = torch.mean(h, dim=1)
        pooled = torch.cat([max_pooled, mean_pooled], dim=-1)
        pooled = self.activation(self.linear(pooled))
        logits = self.classifier(pooled)
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:  # We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        if self.config.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.mem_ind = self.mem_ind.to(*args, **kwargs)
        return self


class TowerLayer(nn.Module):
    # k, v: [mem_tokens; encoder_tokens]
    # q: mem_tokens
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hiddens, encoder_hiddens, encoder_attention_mask):
        tower_att_mask = torch.ones(hiddens.shape[:2], dtype=torch.int64)[:, None, None, :]
        tower_att_mask = tower_att_mask.to(hiddens.device)
        if encoder_attention_mask.dim() == 2:
            encoder_attention_mask = encoder_attention_mask[:, None, None, :]
        attention_to = torch.cat([hiddens, encoder_hiddens], dim=1)
        attention_mask = torch.cat([tower_att_mask, encoder_attention_mask], dim=-1)

        cross_attention_outputs = self.attention(hiddens, None, None, attention_to, attention_mask)

        attention_output = cross_attention_outputs[0]
        outputs = cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertWithMemTower(nn.Module):
    def __init__(self, base_model, num_labels, mem_size=5, output_hidden_states=False, output_attentions=False):
        super().__init__()
        assert output_hidden_states
        self.bert = BertModel.from_pretrained(base_model, output_hidden_states=output_hidden_states,
                                              output_attentions=output_attentions)
        self.mem_tower = MemTower(self.bert.config, num_labels, mem_size)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        assert len(bert_output) >= 3
        assert len(bert_output[2]) == self.bert.config.num_hidden_layers + 1  # +1 for emb output
        layers_hidden = bert_output[2][1:]
        output = self.mem_tower(layers_hidden, attention_mask, labels=labels)
        return output

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.mem_tower = self.mem_tower.to(*args, **kwargs)
        return self


@register('torch_bert_classifier_mt')
class TorchBertClassifierModelMT(TorchModel):
    """Bert-based model for text classification on PyTorch.

    It uses output from [CLS] token and predicts labels using linear transformation.

    Args:
        n_classes: number of classes
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        one_hot_labels: set True if one-hot encoding for labels is used
        multilabel: set True if it is multi-label classification
        return_probas: set True if return class probabilites instead of most probable label needed
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        optimizer: optimizer name from `torch.optim`
        optimizer_parameters: dictionary with optimizer's parameters,
                              e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
        clip_norm: clip gradients by norm coefficient
        bert_config_file: path to Bert configuration file (not used if pretrained_bert is key title)
    """

    def __init__(self, n_classes,
                 pretrained_bert,
                 one_hot_labels: bool = False,
                 multilabel: bool = False,
                 return_probas: bool = False,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: dict = {"lr": 1e-3, "weight_decay": 0.01, "betas": (0.9, 0.999), "eps": 1e-6},
                 clip_norm: Optional[float] = None,
                 bert_config_file: Optional[str] = None,
                 vocab_size: Optional[int] = None,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False,
                 pool_mem_tokens: bool = False,
                 mem_size: Optional[int] = 0,
                 only_head: bool = False,
                 random_init: bool = False,
                 mean_max_pool: bool = False,
                 init_tower_layer_with_encoder: bool = False,
                 **kwargs) -> None:

        self.return_probas = return_probas
        self.one_hot_labels = one_hot_labels
        self.multilabel = multilabel
        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file
        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        # todo: if n_classes == 1 solve regression task (HF sequence classification models have such logic)
        # todo: add normalization preprocessor and postprocessor
        self.n_classes = n_classes
        self.clip_norm = clip_norm
        self.vocab_size = vocab_size
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        # expr args:
        self.pool_mem_tokens = pool_mem_tokens
        self.mem_size = mem_size
        self.only_head = only_head
        self.random_init = random_init
        self.mean_max_pool = mean_max_pool
        self.init_tower_layer_with_encoder = init_tower_layer_with_encoder

        if self.multilabel and not self.one_hot_labels:
            raise RuntimeError('Use one-hot encoded labels for multilabel classification!')

        if self.multilabel and not self.return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

        if self.return_probas and self.n_classes == 1:
            raise RuntimeError('Set return_probas to False for regression task!')

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         **kwargs)

    def train_on_batch(self, features: List[InputFeatures], y: Union[List[int], List[List[int]]]) -> Dict:
        """Train model on given batch.
        This method calls train_op using features and y (labels).

        Args:
            features: batch of InputFeatures
            y: batch of labels (class id or one-hot encoding)

        Returns:
            dict with loss and learning_rate values
        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.attention_mask for f in features]

        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)
        if self.n_classes > 1:
            b_labels = torch.from_numpy(np.array(y)).to(self.device)
        else:
            b_labels = torch.from_numpy(np.array(y, dtype=np.float32)).to(self.device)

        self.optimizer.zero_grad()

        # todo: fix token_type_ids usage
        outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks,
                             labels=b_labels)
        loss = outputs[0]
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    def __call__(self, features: List[InputFeatures]) -> Union[List[int], List[List[float]]]:
        """Make prediction for given features (texts).

        Args:
            features: batch of InputFeatures

        Returns:
            predicted classes or probabilities of each class

        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.attention_mask for f in features]

        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            # todo: fix token_type_ids usage
            output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks)
            logits = output[0]

        if self.return_probas:
            if not self.multilabel:
                pred = torch.nn.functional.softmax(logits, dim=-1)
            else:
                pred = torch.nn.functional.sigmoid(logits)
            pred = pred.detach().cpu().numpy()
        elif self.n_classes > 1:
            logits = logits.detach().cpu().numpy()
            pred = np.argmax(logits, axis=1)
        else:  # regression
            pred = logits.squeeze(-1).detach().cpu().numpy()

        if self.output_attentions:
            # output_hiddens is always true (we need them in MemTower)
            # todo: output_attentions and output_hidden_states for both bert and tower
            return pred, output[1:]

        return pred

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        self.model = BertWithMemTower(base_model=self.pretrained_bert,
                                      num_labels=self.n_classes, mem_size=self.mem_size,
                                      output_hidden_states=self.output_hidden_states,
                                      output_attentions=self.output_attentions)

        if self.init_tower_layer_with_encoder:
            encoder_last_layer_p = dict(self.model.bert.encoder.layer[-1].named_parameters())
            tower_layer_p = self.model.mem_tower.tower_layer.named_parameters()
            for p_name, p in tower_layer_p:
                p.data.copy_(encoder_last_layer_p[p_name].data)

        self.model.to(self.device)

        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        if self.load_path:
            log.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                log.info(f"Load path {weights_path} exists.")
                log.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                log.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.info(f"Init from scratch. Load path {weights_path} does not exist.")
