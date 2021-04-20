from typing import Dict, Any, List

import srsly
from allennlp.common.file_utils import cached_path
from allennlp.data import DatasetReader, TokenIndexer, Token, Instance, Field
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import TextField, SpanField, ListField, LabelField, MetadataField, MultiLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from overrides import overrides

from mare.label.extraction import extract_entities, extract_relations_from_smart_sample


@DatasetReader.register("span_based_smart")
class SpanBasedSmartReader(DatasetReader):
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_namespace: str = "ner_labels",
                 tag_label: str = "ner",
                 include_trigger: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace
        self.tag_label = tag_label
        self.include_trigger = include_trigger

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as smart_file:
            dataset = (srsly.json_loads(line) for line in smart_file.readlines())

            for sample in dataset:
                text = sample["text"]
                words = [text[t["span"]["start"]: t["span"]["end"]] for t in sample["tokens"]]
                tokens = [Token(w) for w in words]

                if self.tag_label == "ner":
                    entities = extract_entities(sample)
                elif self.tag_label == "relation":
                    relations = extract_relations_from_smart_sample(sample, include_trigger=self.include_trigger)
                    entities = [e for relation in relations for e in relation.entities]

                yield self.text_to_instance(tokens,
                                            entities=entities,
                                            relations=relations)

    def text_to_instance(self,
                         tokens: List[Token],
                         entities: List = None,
                         relations: List = None) -> Instance:
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        words = [x.text for x in tokens]
        spans = []
        for start, end in enumerate_spans(words, max_span_width=self._max_span_width):
            assert start >= 0
            assert end >= 0
            spans.append(SpanField(start, end, sequence))


        span_field = ListField(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]
        instance_fields["spans"] = span_field

        ner_labels = [[] for i in span_tuples]

        ner_list = [((e.start, e.end), e.role) for e in entities]

        for span, label in ner_list:
            if self._too_long(span):
                continue
            ix = span_tuples.index(span)
            # if "" in ner_labels[ix]:
            #     ner_labels[ix].remove("")

            ner_labels[ix] += [label]

        instance_fields["ner_labels"] = ListField(
            [MultiLabelField(entry, label_namespace=self.label_namespace) for entry in ner_labels])

        metadata = {"words": words, "relations": relations}
        instance_fields["metadata"] = MetadataField(metadata)

        return Instance(instance_fields)

    def _too_long(self, span):
        return span[1] - span[0] + 1 > self._max_span_width
