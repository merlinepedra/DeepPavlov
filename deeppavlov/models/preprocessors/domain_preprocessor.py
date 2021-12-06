import json
from logging import getLogger
import numpy as np
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('domain_preprocessor')
class DomainPreprocessor(Component):
    def __init__(self, domain_to_ind_filename, domain_topics_filename, *args, **kwargs):
        domain_to_ind_filename = str(expand_path(domain_to_ind_filename))
        domain_topics_filename = str(expand_path(domain_topics_filename))
        with open(domain_to_ind_filename, 'r') as fl:
            self.domain_to_ind = json.load(fl)
        topics_list = []
        self.n_to_domain = {}
        with open(domain_topics_filename, 'r') as fl:
            domain_topics = json.load(fl)
        for n, (domain, topics) in enumerate(list(domain_topics.items())):
            topics_list.append(topics)
            self.n_to_domain[n] = domain
        self.topics_matrix = np.array(topics_list)
    
    def __call__(self, sources, targets, sources_topics, targets_topics):
        source_ids, target_ids, target_known_batch = [], [], []
        for source, target, source_topic, target_topic in \
                zip(sources, targets, sources_topics, targets_topics):
            target_known = True
            if source in self.domain_to_ind:
                source_id = self.domain_to_ind[source]
            elif source_topic:
                dot_products = np.sum(self.topics_matrix * np.array([source_topic]))
                max_domain_n = np.argmax(dot_products)
                max_domain = self.n_to_domain[max_domain_n]
                source_id = self.domain_to_ind[max_domain]
            else:
                source_id = 0
                target_known = False
            
            if target in self.domain_to_ind:
                target_id = self.domain_to_ind[target]
            elif target_topic:
                dot_products = np.sum(self.topics_matrix * np.array([target_topic]))
                max_domain_n = np.argmax(dot_products)
                max_domain = self.n_to_domain[max_domain_n]
                target_id = self.domain_to_ind[max_domain]
            else:
                target_id = 0
                target_known = False
            target_known_batch.append(target_known)
            
            source_ids.append(source_id)
            target_ids.append(target_id)
        
        return source_ids, target_ids, target_known_batch
