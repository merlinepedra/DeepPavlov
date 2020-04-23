from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)


@register('odqa_kbqa_chooser')
class ODKBQAchooser(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, odqa_answers, odqa_answer_scores, kbqa_answers, kbqa_answer_scores):
        answers = []
        for odqa_answer, odqa_answer_score, kbqa_answer, kbqa_answer_score in \
                zip(odqa_answers, odqa_answer_scores, kbqa_answers, kbqa_answer_scores):

            answer = kbqa_answer if kbqa_answer_score > 0.8 else odqa_answer
            answers.append(answer)
        log.debug(f"odqa_answers: {odqa_answers}, {odqa_answer_scores}")
        log.debug(f"kbqa_answers: {kbqa_answers}, {kbqa_answer_scores}")

        return answers
