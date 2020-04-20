from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

log = getLogger(__name__)

@register('doc_ids_concat')
class DocIdsConcat(Component):
    def __init__(self, doc_ids_to_leave: int = 5, *args, **kwargs):
        self.doc_ids_to_leave = doc_ids_to_leave

    def __call__(self, doc_ids_from_ranker_batch, doc_ids_from_linker_batch):
        doc_ids_batch = []
        for doc_ids_from_ranker, doc_ids_from_linker in zip(doc_ids_from_ranker_batch, doc_ids_from_linker_batch):
            doc_ids = [doc_id for ids in doc_ids_from_linker for doc_id in ids[:self.doc_ids_to_leave]] + doc_ids_from_ranker
            doc_ids_batch.append(doc_ids)
        log.debug(f"doc_ids: {doc_ids_batch}")

        return doc_ids_batch
