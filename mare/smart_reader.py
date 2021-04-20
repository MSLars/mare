import itertools
import logging
from abc import ABC
from typing import Dict, Iterable, List, Sequence

import srsly
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Instance, Field, Token
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from mare.label.extraction import extract_relations_from_smart_sample, combined_relation_tags, extract_entities, \
    combine_spans_to_entity_tags

logger = logging.getLogger(__name__)


@DatasetReader.register("smart_single")
class SmartReader(DatasetReader, ABC):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "relation",
                 feature_labels: Sequence[str] = (),
                 coding_scheme: str = "BIO",
                 label_namespace: str = "labels",
                 include_mode: bool = False,
                 include_trigger: bool = True,
                 **kwargs, ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace
        self._original_coding_scheme = "BIO"
        self._include_mode = include_mode
        self.include_trigger = include_trigger

    def _read(self, file_path: str) -> Iterable[Instance]:

        file_path = cached_path(file_path)

        with open(file_path, "r") as smart_file:
            logger.info("Reading instances from lines in file at: %s", file_path)

            dataset = (srsly.json_loads(line) for line in smart_file.readlines())

            for sample in dataset:
                text = sample["text"]
                words = [text[t["span"]["start"]: t["span"]["end"]] for t in sample["tokens"]]
                tokens = [Token(w) for w in words]

                entities = extract_entities(sample)
                entity_tags = combine_spans_to_entity_tags(entities, len(words))

                relations = extract_relations_from_smart_sample(sample, only_mandatory=False,
                                                                include_trigger=self.include_trigger)
                relation_tags = combined_relation_tags(relations, len(words), include_mode=self._include_mode, )

                idx = sample["id"]
                yield self.text_to_instance(tokens,
                                            relation_tags=relation_tags,
                                            relations=relations,
                                            entity_tags=entity_tags,
                                            entities=entities,
                                            idx=idx)

    def text_to_instance(self,
                         tokens: List[Token],
                         relation_tags: List[str] = None,
                         relations: List[Dict] = None,
                         entity_tags: List[str] = None,
                         entities: List = None,
                         idx: str = None) -> Instance:

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}

        metadata = {"words": [x.text for x in tokens],
                    "id": idx}

        if relation_tags is not None:
            instance_fields["relation_tags"] = SequenceLabelField(relation_tags, sequence, "relation_tags")
        if entity_tags is not None:
            instance_fields["entity_tags"] = SequenceLabelField(relation_tags, sequence, "entity_tags")
        if relations is not None:
            metadata["relations"] = relations
        if entities is not None:
            metadata["entities"] = entities

        instance_fields["metadata"] = MetadataField(metadata)

        if self.tag_label == "ner":
            if entity_tags is not None:
                instance_fields["tags"] = SequenceLabelField(entity_tags, sequence, self.label_namespace)
        else:
            if relation_tags is not None:
                instance_fields["tags"] = SequenceLabelField(relation_tags, sequence, self.label_namespace)

        return Instance(instance_fields)
