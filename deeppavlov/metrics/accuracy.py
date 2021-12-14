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


import itertools
from typing import List, Iterable

import numpy as np
import pymorphy2

from deeppavlov.core.common.metrics_registry import register_metric
from deeppavlov.models.go_bot.nlg.dto.json_nlg_response import JSONNLGResponse


morph = pymorphy2.MorphAnalyzer()

@register_metric('accuracy')
def accuracy(y_true: [list, np.ndarray], y_predicted: [list, np.ndarray]) -> float:
    """
    Calculate accuracy in terms of absolute coincidence

    Args:
        y_true: array of true values
        y_predicted: array of predicted values

    Returns:
        fraction of absolutely coincidental samples
    """
    examples_len = len(y_true)
    # if y1 and y2 are both arrays, == can be erroneously interpreted as element-wise equality

    def _are_equal(y1, y2):
        answer = (y1 == y2)
        if isinstance(answer, np.ndarray):
            answer = answer.all()
        return answer

    equalities = [_are_equal(y1, y2) for y1, y2 in zip(y_true, y_predicted)]
    correct = sum(equalities)
    return correct / examples_len if examples_len else 0


def domain_acc_func(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted) -> float:
    num_correct = 0
    class_pred_list, class_true_list = [], []
    for (class_pred, topic_pred, token_pred, sent_pred), class_true, topic_true, token_true in \
            zip(y_predicted, class_true_batch, topic_true_batch, token_true_batch):
        class_pred_list.append(class_pred)
        class_true_list.append(class_true)
        if class_pred == class_true:
            num_correct += 1
    num_total = len(y_predicted)
    return round(num_correct / num_total, 4)


def topic_acc_func(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted) -> float:
    num_correct, num_pred, num_total = 0, 0, 0
    for (class_pred, topic_pred, token_pred, sent_pred), class_true, topic_true, token_true in \
            zip(y_predicted, class_true_batch, topic_true_batch, token_true_batch):
        if int(class_pred) == 1:
            num_correct += len(set(topic_pred).intersection(set(topic_true)))
            num_total += len(topic_true)
            num_pred += len(topic_pred)
    
    if num_pred > 0:
        precision = num_correct / num_pred
    else:
        precision = 0.0
    if num_total > 0:
        recall = num_correct / num_total
    else:
        recall = 0.0
    if precision + recall > 0.0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return round(f1, 4)


def token_acc_func(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted) -> float:
    num_correct, num_pred, num_total = 0, 0, 0
    for (class_pred, topic_pred, token_pred, sent_pred), class_true, topic_true, token_true in \
            zip(y_predicted, class_true_batch, topic_true_batch, token_true_batch):
        if int(class_pred) == 1:
            token_pred = set([morph.parse(tok.lower())[0].normal_form for tok in token_pred])
            token_true = set([morph.parse(tok.lower())[0].normal_form for tok in token_true])
            num_correct += len(token_pred.intersection(token_true))
            num_total += len(token_true)
            num_pred += len(token_pred)
    
    if num_pred > 0:
        precision = num_correct / num_pred
    else:
        precision = 0
    if num_total > 0:
        recall = num_correct / num_total
    else:
        recall = 0
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return round(f1, 4)


@register_metric('domain_acc')
def domain_acc(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted):
    return domain_acc_func(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted)


@register_metric('topic_acc')
def topic_acc(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted):
    return topic_acc_func(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted)


@register_metric('token_acc')
def token_acc(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted):
    return token_acc_func(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted)


@register_metric('domain_acc2')
def domain_acc2(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch,
                topic_labels, token_labels, y_predicted):
    return domain_acc_func(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted)


@register_metric('topic_acc2')
def topic_acc2(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch,
               topic_labels, token_labels, y_predicted):
    return topic_acc_func(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted)


@register_metric('token_acc2')
def token_acc2(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch,
               topic_labels, token_labels, y_predicted):
    return token_acc_func(class_true_batch, token_true_batch, topic_true_batch, sentiment_batch, y_predicted)


@register_metric('multitask_accuracy')
def multitask_accuracy(*args) -> float:
    """
    Accuracy for multiple simultaneous tasks.

    Args:
        *args: a list of `2n` inputs. The first `n` inputs are the correct answers for `n` tasks,
            and the last `n` are the predicted ones.

    Returns:
        The percentage of inputs where the answers for all `n` tasks are correct.
    """
    n = len(args)
    y_true_by_tasks, y_predicted_by_tasks = args[:n // 2], args[n // 2:]
    y_true, y_predicted = list(zip(*y_true_by_tasks)), list(zip(*y_predicted_by_tasks))
    return accuracy(y_true, y_predicted)


@register_metric('multitask_sequence_accuracy')
def multitask_sequence_accuracy(*args) -> float:
    """
    Accuracy for multiple simultaneous sequence labeling (tagging) tasks.
    For each sequence the model checks whether all its elements
    are labeled correctly for all the individual taggers.

    Args:
        *args: a list of `2n` inputs. The first `n` inputs are the correct answers for `n` tasks,
            and the last `n` are the predicted ones. For each task an

    Returns:
        The percentage of sequences where all the items has correct answers for all `n` tasks.

    """
    n = len(args)
    y_true_by_tasks, y_predicted_by_tasks = args[:n // 2], args[n // 2:]
    y_true_by_sents = list(zip(*y_true_by_tasks))
    y_predicted_by_sents = list(zip(*y_predicted_by_tasks))
    y_true = list(list(zip(*elem)) for elem in y_true_by_sents)
    y_predicted = list(list(zip(*elem)) for elem in y_predicted_by_sents)
    return accuracy(y_true, y_predicted)


@register_metric('multitask_token_accuracy')
def multitask_token_accuracy(*args) -> float:
    """
        Per-item accuracy for multiple simultaneous sequence labeling (tagging) tasks.

        Args:
            *args: a list of `2n` inputs. The first `n` inputs are the correct answers for `n` tasks
                and the last `n` are the predicted ones. For each task an

        Returns:
            The percentage of sequence elements for which the answers for all `n` tasks are correct.

        """
    n = len(args)
    y_true_by_tasks, y_predicted_by_tasks = args[:n // 2], args[n // 2:]
    y_true_by_sents = list(zip(*y_true_by_tasks))
    y_predicted_by_sents = list(zip(*y_predicted_by_tasks))
    y_true = list(list(zip(*elem)) for elem in y_true_by_sents)
    y_predicted = list(list(zip(*elem)) for elem in y_predicted_by_sents)
    return per_token_accuracy(y_true, y_predicted)


@register_metric('sets_accuracy')
def sets_accuracy(y_true: [list, np.ndarray], y_predicted: [list, np.ndarray]) -> float:
    """
    Calculate accuracy in terms of sets coincidence

    Args:
        y_true: true values
        y_predicted: predicted values

    Returns:
        portion of samples with absolutely coincidental sets of predicted values
    """
    examples_len = len(y_true)
    correct = sum([set(y1) == set(y2) for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


@register_metric('slots_accuracy')
def slots_accuracy(y_true, y_predicted):
    y_true = [{tag.split('-')[-1] for tag in s if tag != 'O'} for s in y_true]
    y_predicted = [set(s.keys()) for s in y_predicted]
    return accuracy(y_true, y_predicted)


@register_metric('per_token_accuracy')
def per_token_accuracy(y_true, y_predicted):
    y_true = list(itertools.chain(*y_true))
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


# region go-bot metrics

@register_metric('per_item_dialog_accuracy')
def per_item_dialog_accuracy(y_true, y_predicted: List[List[str]]):
    # todo metric classes???
    y_true = [y['text'] for dialog in y_true for y in dialog]
    y_predicted = itertools.chain(*y_predicted)
    examples_len = len(y_true)
    correct = sum([y1.strip().lower() == y2.strip().lower() for y1, y2 in zip(y_true, y_predicted)])
    return correct / examples_len if examples_len else 0


@register_metric("per_item_action_accuracy")
def per_item_action_accuracy(dialogs_true, dialog_jsons_predicted: List[List[JSONNLGResponse]]):
    # todo metric classes???
    # todo oop instead of serialization/deserialization
    utterances_actions_true = [utterance['act']
                               for dialog in dialogs_true
                               for utterance in dialog]

    utterances_actions_predicted: Iterable[JSONNLGResponse] = itertools.chain(*dialog_jsons_predicted)
    examples_len = len(utterances_actions_true)
    correct = sum([y1.strip().lower() == '+'.join(y2.actions_tuple).lower()
                   for y1, y2 in zip(utterances_actions_true, utterances_actions_predicted)])  # todo ugly
    return correct / examples_len if examples_len else 0

# endregion go-bot metrics


@register_metric('acc')
def round_accuracy(y_true, y_predicted):
    """
    Rounds predictions and calculates accuracy in terms of absolute coincidence.

    Args:
        y_true: list of true values
        y_predicted: list of predicted values

    Returns:
        portion of absolutely coincidental samples
    """
    if isinstance(y_predicted[0], np.ndarray):
        predictions = [np.round(x) for x in y_predicted]
    else:
        predictions = [round(x) for x in y_predicted]
    examples_len = len(y_true)
    correct = sum([y1 == y2 for y1, y2 in zip(y_true, predictions)])
    return correct / examples_len if examples_len else 0


@register_metric('kbqa_accuracy')
def kbqa_accuracy(y_true, y_predicted):
    total_correct = 0
    for answer_true, answer_predicted in zip(y_true, y_predicted):
        if answer_predicted in answer_true:
            total_correct += 1

    return total_correct / len(y_true) if len(y_true) else 0
