import json
from logging import getLogger
from pathlib import Path
from typing import List, Union

from deeppavlov import Chainer
from deeppavlov.models.go_bot.nlu.dto.nlu_response import NLUResponse
from deeppavlov.models.go_bot.nlu.nlu_manager_interface import NLUManagerInterface

log = getLogger(__name__)


# todo add the ability to configure nlu loglevel in config (now the setting is shared across all the GO-bot)
# todo add each method input-output logging when proper loglevel level specified


class NLUManager(NLUManagerInterface):
    """
    NLUManager is a unit of the go-bot pipeline that handles the understanding of text.
    Given the text it provides tokenization, intents extraction and the slots extraction.
    (the whole go-bot pipeline is as follows: NLU, dialogue-state-tracking&policy-NN, NLG)
    """

    def __init__(self, tokenizer, slot_filler, intent_classifier, debug=False, **kwargs):
        self.debug = debug
        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} init(): "
                      f"tokenizer={tokenizer}, slot_filler={slot_filler}, "
                      f"intent_classifier={intent_classifier}, debug={debug}")
        # todo type hints
        self.tokenizer = tokenizer
        self.slot_filler = slot_filler
        self.intent_classifier = intent_classifier
        self.intents = []
        if isinstance(self.intent_classifier, Chainer):
            self.intents = self.intent_classifier.get_main_component().classes

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} init(): "
                      f"tokenizer={tokenizer}, slot_filler={slot_filler}, "
                      f"intent_classifier={intent_classifier}, debug={debug}")

    def nlu(self, text: str) -> NLUResponse:
        """
        Extracts slot values and intents from text.

        Args:
            text: text to extract knowledge from

        Returns:
            an object storing the extracted slos and intents info
        """
        # todo meaningful type hints
        tokens = self._tokenize_single_text_entry(text)

        slots = None
        if callable(self.slot_filler):
            slots = self._extract_slots_from_tokenized_text_entry(tokens)

        intents = []
        if callable(self.intent_classifier):
            intents = self._extract_intents_from_tokenized_text_entry(tokens)

        return NLUResponse(slots, intents, tokens)

    def _extract_intents_from_tokenized_text_entry(self, tokens: List[str]):
        # todo meaningful type hints, relies on unannotated intent classifier
        intent_features = self.intent_classifier([' '.join(tokens)])[1][0]
        return intent_features

    def _extract_slots_from_tokenized_text_entry(self, tokens: List[str]):
        # todo meaningful type hints, relies on unannotated slot filler
        return self.slot_filler([tokens])[0]

    def _tokenize_single_text_entry(self, text: str):
        # todo meaningful type hints, relies on unannotated tokenizer
        return self.tokenizer([text.lower().strip()])[0]

    def num_of_known_intents(self) -> int:
        """
        Returns:
            the number of intents known to the NLU module
        """
        return len(self.intents)


class MockNLUManager(NLUManager):

    def __init__(self, tokenizer, slot_filler, intent_classifier, dataset_folder, **kwargs):
        super().__init__(tokenizer, slot_filler, intent_classifier)

        dataset = json.load(open(f"{dataset_folder}/simple-dstc2-trn.json")) + \
                  json.load(open(f"{dataset_folder}/simple-dstc2-tst.json")) + \
                  json.load(open(f"{dataset_folder}/simple-dstc2-val.json"))
        user_turns = [turn
                      for dialogue in dataset
                      for turn in dialogue
                      if turn.get("speaker", '') == 1]

        self.texts2textsids = {}
        # known_intents = set()
        from collections import defaultdict
        self.textsids2intents = defaultdict(list)
        self.textsids2slots = {}
        for turn_ix, turn in enumerate(user_turns):
            self.texts2textsids[turn["text"]] = turn_ix
            self.textsids2slots[turn_ix] = turn.get("slots", [])
            self.textsids2intents[turn_ix] = [act
                                              for act in turn.get("act", '').split('+')
                                              if act != '']

        self.known_intents = sorted(set(intent
            for intents_group in self.textsids2intents.values()
            for intent in intents_group
        ))


        self.textsids2nlu = {}
        for turn in user_turns:
            text_ix = self.texts2textsids[turn["text"]]
            tokens = self._tokenize_single_text_entry(turn["text"])
            slots = dict(self.textsids2slots.get(text_ix, []))

            intents_mhe = [0.] * len(self.known_intents)
            for action in self.textsids2intents[turn_ix]:
                intents_mhe[self.known_intents.index(action)] = 1.

            self.textsids2nlu[text_ix] = NLUResponse(slots, intents_mhe, tokens)

    def extract_nlu(self, turn) -> NLUResponse:
        text_ix = self.texts2textsids[turn["text"]]
        return self.textsids2nlu[text_ix]

    def nlu(self, text: str) -> NLUResponse:
        nlu = self.textsids2nlu.get(self.texts2textsids.get(text, -1), NLUResponse(dict,
                                                                                   [0.] * len(self.known_intents),
                                                                                   self._tokenize_single_text_entry(text)))
        return nlu

    def num_of_known_intents(self) -> int:
        return len(self.known_intents)
