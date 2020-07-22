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
from typing import List, Dict, Union
import numpy as np

import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers.data.processors.utils import InputFeatures

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

logger = getLogger(__name__)


@register('torch_bert_classifier')
class TorchBertClassifierModel(TorchModel):
    """Bert-based model for text classification on PyTorch.

    It uses output from [CLS] token and predicts labels using linear transformation.

    Args:
        bert_config_file: path to Bert configuration file
        n_classes: number of classes
        keep_prob: dropout keep_prob for non-Bert layers
        one_hot_labels: set True if one-hot encoding for labels is used
        multilabel: set True if it is multi-label classification
        return_probas: set True if return class probabilites instead of most probable label needed
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        optimizer: name of tf.train.* optimizer or None for `AdamWeightDecayOptimizer`
        num_warmup_steps:
        weight_decay_rate: L2 weight decay for `AdamWeightDecayOptimizer`
        pretrained_bert: pretrained Bert checkpoint
        min_learning_rate: min value of learning rate if learning rate decay is used
    """

    def __init__(self, bert_config_file, n_classes, keep_prob,
                 one_hot_labels=False, multilabel=False, return_probas=False,
                 attention_probs_keep_prob=None, hidden_keep_prob=None,
                 optimizer=None, num_warmup_steps=None, weight_decay_rate=0.01,
                 pretrained_bert=None, min_learning_rate=1e-06, **kwargs) -> None:
        super().__init__(**kwargs)

        self.return_probas = return_probas
        self.n_classes = n_classes
        self.min_learning_rate = min_learning_rate
        self.keep_prob = keep_prob
        self.one_hot_labels = one_hot_labels
        self.multilabel = multilabel
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.weight_decay_rate = weight_decay_rate

        if self.multilabel and not self.one_hot_labels:
            raise RuntimeError('Use one-hot encoded labels for multilabel classification!')

        if self.multilabel and not self.return_probas:
            raise RuntimeError('Set return_probas to True for multilabel classification!')

        self.bert_config = BertConfig.from_json_file(str(expand_path(bert_config_file)))

        if attention_probs_keep_prob is not None:
            self.bert_config.attention_probs_dropout_prob = 1.0 - attention_probs_keep_prob
        if hidden_keep_prob is not None:
            self.bert_config.hidden_dropout_prob = 1.0 - hidden_keep_prob

        if pretrained_bert:
            self.bert = BertForSequenceClassification.from_pretrained(pretrained_bert, num_labels=n_classes)
                # tutorial has this PARAMS also
                # output_attentions=False,  # Whether the model returns attentions weights.
                # output_hidden_states=False,  # Whether the model returns all hidden-states.
        else:
            self.bert = BertForSequenceClassification(config=self.bert_config)

        self.optimizer = AdamW(self.bert.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay_rate,
                               betas=(0.9, 0.999), eps=1e-6)
        # exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"] < --- было в оптимайзере что это?

        if self.load_path is not None:
            self.load()
        self.bert.to(self.device)
        # need to move it to `eval` mode because it can be used in `build_model` (not by `torch_trainer`
        self.bert.eval()

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
        input_type_ids = [f.token_type_ids for f in features]

        b_input_ids = torch.from_numpy(input_ids).to(self.device)
        b_input_mask = torch.from_numpy(input_masks).to(self.device)
        b_input_type_ids = torch.from_numpy(input_type_ids).to(self.device)

        b_labels = torch.from_numpy(y).to(self.device)

        self.optimizer.zero_grad()

        loss, logits = self.bert(b_input_ids, token_type_ids=b_input_type_ids, attention_mask=b_input_mask,
                                 labels=b_labels)
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(self.bert.parameters(), 1.0)

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
        input_type_ids = [f.token_type_ids for f in features]

        b_input_ids = torch.from_numpy(input_ids).to(self.device)
        b_input_mask = torch.from_numpy(input_masks).to(self.device)
        b_input_type_ids = torch.from_numpy(input_type_ids).to(self.device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = self.bert(b_input_ids, token_type_ids=b_input_type_ids, attention_mask=b_input_mask)

        logits = outputs[0]
        # Move logits and labels to CPU and to numpy arrays
        logits = logits.detach().cpu().numpy()

        if not self.return_probas:
            pred = logits
        else:
            pred = [np.argmax(vec, axis=1).flatten() for vec in logits]
        return pred