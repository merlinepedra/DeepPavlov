import json
from itertools import combinations
from pathlib import Path
from typing import Union, Dict, List, Tuple, NewType, Type

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register, get_model
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
from deeppavlov.dataset_readers.md_yaml_dialogs_reader import MD_YAML_DialogsDatasetReader
from deeppavlov.models.go_bot.dto.dataset_features import BatchDialoguesFeatures
from deeppavlov.models.go_bot.nlg.dto.json_nlg_response import JSONNLGResponse, VerboseJSONNLGResponse
from deeppavlov.models.go_bot.nlg.nlg_manager import log
from deeppavlov.models.go_bot.nlg.nlg_manager_interface import NLGManagerInterface
from deeppavlov.models.go_bot.policy.dto.policy_prediction import PolicyPrediction
import deeppavlov.models.go_bot.nlg.templates.templates as go_bot_templates

import re

NLGCompatibleDatasetReader = NewType("NLGCompatibleDatasetReader",
                                     MD_YAML_DialogsDatasetReader)
@register("gobot_json_nlg_manager")
class MockJSONNLGManager(NLGManagerInterface):

    # todo inheritance
    # todo force a2id, id2a mapping to be persistent for same configs

    def __init__(self,
                 actions2slots_path: Union[str, Path],
                 api_call_action: str,
                 data_path: Union[str, Path],
                 dataset_reader_class="dstc2_reader",
                 debug=False):
        self.debug = debug

        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} init(): "
                      f"actions2slots_path={actions2slots_path}, "
                      f"api_call_action={api_call_action}, debug={debug}")

        # noinspection PyTypeChecker
        self._dataset_reader: Type[NLGCompatibleDatasetReader] \
            = get_model(dataset_reader_class)

        individual_actions2slots = self._load_actions2slots_mapping(actions2slots_path)
        possible_actions_combinations_tuples = sorted(
            set(actions_combination_tuple
                for actions_combination_tuple
                in self._extract_actions_combinations(data_path)),
            key=lambda x: '+'.join(x))
        self.templates = self._load_action_templates(data_path)

        self.action_tuples2ids = {action_tuple: action_tuple_idx
                                  for action_tuple_idx, action_tuple
                                  in enumerate(possible_actions_combinations_tuples)}  # todo: typehint tuples somehow
        self.ids2action_tuples = {v: k for k, v in self.action_tuples2ids.items()}

        self.action_tuples_ids2slots = {}  # todo: typehint tuples somehow
        for actions_combination_tuple in possible_actions_combinations_tuples:
            actions_combination_slots = set(slot
                                            for action in actions_combination_tuple
                                            for slot in individual_actions2slots.get(action, []))
            actions_combination_tuple_id = self.action_tuples2ids[actions_combination_tuple]
            self.action_tuples_ids2slots[actions_combination_tuple_id] = actions_combination_slots

        self._api_call_id = -1
        if api_call_action is not None:
            api_call_action_as_tuple = (api_call_action,)
            self._api_call_id = self.action_tuples2ids[api_call_action_as_tuple]

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} init(): "
                      f"actions2slots_path={actions2slots_path}, "
                      f"api_call_action={api_call_action}, debug={debug}")

    def get_api_call_action_id(self) -> int:
        """
        Returns:
            an ID corresponding to the api call action
        """
        return self._api_call_id

    def _extract_actions_combinations(self, dataset_path: Union[str, Path]):
        dataset_path = expand_path(dataset_path)
        dataset = self._dataset_reader.read(data_path=dataset_path, dialogs=True, ignore_slots=True)
        actions_combinations = set()
        for dataset_split in dataset.values():
            for dialogue in dataset_split:
                for user_input, system_response in dialogue:
                    actions_tuple = tuple(system_response["act"].split('+'))
                    actions_combinations.add(actions_tuple)
        return actions_combinations

    def _load_action_templates(self, dataset_path: Union[str, Path]):
        dataset_path = expand_path(dataset_path)

        # region code duplication; replace w dataset = self._dataset_reader.read
        domain_fname = self._dataset_reader.DOMAIN_FNAME
        nlu_fname = self._dataset_reader.NLU_FNAME
        stories_fnames = tuple(self._dataset_reader._data_fname(dt)
                               for dt in self._dataset_reader.VALID_DATATYPES)
        required_fnames = stories_fnames + (nlu_fname, domain_fname)
        for required_fname in required_fnames:
            required_path = Path(dataset_path, required_fname)
            if not required_path.exists():
                log.error(f"INSIDE MockJSONNLGManager._load_action_templates(): "
                          f"{required_fname} not found with path {required_path}")
        domain_path = Path(dataset_path, domain_fname)
        from deeppavlov.dataset_readers.md_yaml_dialogs_reader import DomainKnowledge
        # endregion
        domain_knowledge = DomainKnowledge.from_yaml(domain_path)
        templates: Dict[str, List[str]] = domain_knowledge.response_templates

        template_type = "DefaultTemplate"
        templates_o = go_bot_templates.RandTemplates(template_type)
        for act, templ_li in templates.items():
            for templ in templ_li:
                oldstyle_templ = re.sub(self._dataset_reader._SLOTS_MARKUP_PATTERN,
                                        r"#\g<slot_name>", templ)
                # IMPORTANT: assignment is ok below, it actually adds to set
                templates_o[act] = templates_o.ttype.from_str(oldstyle_templ)
        return templates_o

    @staticmethod
    def _load_actions2slots_mapping(actions2slots_json_path) -> Dict[str, str]:
        actions2slots_json_path = expand_path(actions2slots_json_path)
        if actions2slots_json_path.exists():
            with open(actions2slots_json_path, encoding="utf-8") as actions2slots_json_f:
                actions2slots = json.load(actions2slots_json_f)
        else:
            actions2slots = dict()
            log.info(f"INSIDE {__class__.__name__} _load_actions2slots_mapping(): "
                      f"actions2slots_json_path={actions2slots_json_path} DOES NOT EXIST. "
                      f"initialized actions2slots mapping with an empty one: {str(actions2slots)}")
        return actions2slots

    def get_action_id(self, action_text: Union[str, Tuple[str, ...]]) -> int:
        """
        Looks up for an ID corresponding to the passed action text.

        Args:
            action_text: the text for which an ID needs to be returned.
        Returns:
            an ID corresponding to the passed action text
        """
        if isinstance(action_text, str):
            actions_tuple = tuple(action_text.split('+'))
        else:
            actions_tuple = action_text
        return self.action_tuples2ids[actions_tuple]  # todo unhandled exception when not found

    def decode_response(self,
                        utterance_batch_features: BatchDialoguesFeatures,
                        policy_prediction: PolicyPrediction,
                        tracker_slotfilled_state: dict) -> JSONNLGResponse:
        """
        Converts the go-bot inference objects to the single output object.

        Args:
            utterance_batch_features: utterance features extracted in go-bot that
            policy_prediction: policy model prediction (predicted action)
            tracker_slotfilled_state: tracker knowledge before the NLG is performed

        Returns:
            The NLG output unit that stores slot values and predicted actions info.
        """
        slots_to_log = self.action_tuples_ids2slots[policy_prediction.predicted_action_ix]

        slots_values = {slot_name: tracker_slotfilled_state.get(slot_name, "unk") for slot_name in slots_to_log}
        actions_tuple = self.ids2action_tuples[policy_prediction.predicted_action_ix]
        texts_tuple = (self._generate_slotfilled_text_for_action(a, slots_values)
                       for a in actions_tuple)
        response = JSONNLGResponse(slots_values, actions_tuple)
        verbose_response = VerboseJSONNLGResponse.from_json_nlg_response(response)
        verbose_response.policy_prediction = policy_prediction
        verbose_response.texts = texts_tuple
        return verbose_response

    def _generate_slotfilled_text_for_action(self, action: str, slots: dict) -> str:
        """
        Generate text for the predicted speech action using the pattern provided.
        The slotfilled state provides info to encapsulate to the pattern.

        Args:
            action: the action to generate text for.
            slots: the slots and their known values. usually received from dialogue state tracker.

        Returns:
            the text generated for the passed action and slot values.
        """
        text = self.templates.templates[action].generate_text(slots)
        return text

    def num_of_known_actions(self) -> int:
        """
        Returns:
            the number of actions known to the NLG module
        """
        return len(self.action_tuples2ids.keys())

    def known_actions(self) -> List:
        """
        Returns:
             the list of actions known to the NLG module
        """
        return list(self.action_tuples2ids.keys())
