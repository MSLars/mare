import json
import random

import torch
from overrides import overrides

from allennlp.data import Token, Field, Instance
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import TextField, SpanField, ListField, LabelField, MetadataField, MultiLabelField, ArrayField

import numpy as np

from typing import Any, Dict

from spart.util.util import extract_relations_from_smart_sample, extract_entities

relation_args_names = {
    "Accident": ["location", "trigger"],
    "CanceledRoute": ["location", "trigger"],
    "CanceledStop": ["location", "trigger"],
    "Delay": ["location", "trigger"],
    "Disaster": ["type", "location"],
    "Obstruction": ["location", "trigger"],
    "RailReplacementService": ["location", "trigger"],
    "TrafficJam": ["location", "trigger"],
    "Acquisition": ["buyer", "acquired"],
    "Insolvency": ["company", "trigger"],
    "Layoffs": ["company", "trigger"],
    "Merger": ["old", "old"],
    "OrganizationLeadership": ["organization", "person"],
    "SpinOff": ["parent", "child"],
    "Strike": ["company", "trigger"]
}


def create_mask(start, end, context):
    mask = np.zeros(context)
    mask[start:(end+1)] = 1

    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_mask(start, end, context_size)
    return mask


@DatasetReader.register("spart")
class SpartReader(DatasetReader):
    def __init__(self,
                 max_span_width: int,
                 max_relation_negative_samples: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 training: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._max_relation_negative_samples = max_relation_negative_samples
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        # we cannot
        self._training = training

    @overrides
    def _read(self, file_path: str):

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Loop over the documents.
            doc_text = json.loads(line)

            if len(doc_text["tokens"]) <= 2:
                continue

            instance = self.text_to_instance(doc_text)
            yield instance

    @overrides
    def text_to_instance(self, sample: Dict[str, Any], training: bool = True):
        text = sample["text"]
        words = [text[t["span"]["start"]: t["span"]["end"]] for t in sample["tokens"]]
        tokens = [Token(w) for w in words]
        entities = extract_entities(sample)

        relations = extract_relations_from_smart_sample(sample, include_trigger=True)

        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence}
        words = [x.text for x in tokens]
        spans = []
        span_masks = []

        context_size = len(words) + 1
        for start, end in enumerate_spans(words,
                                          max_span_width=self._max_span_width):  # TODO beim training wird eigentlich keine vollständige candidate liste genommen
            assert start >= 0
            assert end >= 0
            spans.append(SpanField(start, end, sequence))
            span_masks.append(create_mask(start, end, context_size))

        instance_fields["span_masks"] = ListField(
            [ArrayField(np.array(si, dtype=np.int), dtype=np.int) for si in span_masks])

        span_field = ListField(spans)

        span_tuples = [(span.span_start, span.span_end) for span in spans]
        instance_fields["spans"] = span_field  # TODO was ist mit dem negative sampling?

        ner_labels = ["O" for i in span_tuples]

        ner_list = [((e.start, e.end), e.role) for e in entities]

        for span, label in ner_list:
            if self._too_long(span):
                continue
            ix = span_tuples.index(span)
            ner_labels[ix] = label

        # TODO Evaluate if this should be a MultiLabel instead of Label
        instance_fields["ner_labels"] = ListField(
            [LabelField(entry, label_namespace="ner_labels") for entry in ner_labels])

        pos_span_pairs = []
        pos_span_labels = []
        pos_span_masks = []

        for rel in relations:
            mand_arg_roles = relation_args_names[rel.label]
            try:
                # TODO handle special case Merger
                s1, s2 = sorted([s for s in rel.spans if s.role in mand_arg_roles], key=lambda x: x.role)
                pos_span_pairs += [(span_tuples.index(s1.span), span_tuples.index(s2.span))]
                pos_span_labels += [[rel.label]]
                pos_span_masks.append(create_rel_mask((s1.start, s1.end), (s2.start, s2.end), context_size))
            except ValueError:
                pass
            except Exception:
                i = 10

        neg_span_pairs = []
        neg_span_labels = []
        neg_span_masks = []
        if len(ner_list) < 2:
            ner_cands = random.sample(span_tuples, min(len(span_tuples), 7))

            ner_cands = [nc for nc in ner_cands if not self._too_long(nc)]

            ner_list += [(s, "") for s in ner_cands]

        for i1, s1 in enumerate(ner_list):
            for i2, s2 in enumerate(ner_list):
                # rev = (s2, s1)
                # rev_symmetric = rev in pos_rel_spans and pos_rel_types[pos_rel_spans.index(rev)].symmetric
                if self._too_long(s1[0]) or self._too_long(s2[0]):
                    continue
                # candidate
                cand = (span_tuples.index(s1[0]), span_tuples.index(s2[0]))

                # do not add as negative relation sample:
                # neg. relations from an entity to itself
                # entity pairs that are related according to gt
                # entity pairs whose reverse exists as a symmetric relation in gt
                # if s1 != s2 and (s1, s2) not in pos_span_pairs and not rev_symmetric:
                if cand[0] != cand[1] and cand not in pos_span_pairs:
                    neg_span_pairs += [cand]
                    neg_span_labels += [[]]
                    neg_span_masks.append(create_rel_mask(s1[0], s2[0], context_size))

        negative_samples = random.sample(
                list(zip(neg_span_pairs, neg_span_labels, neg_span_masks)),
                min(len(neg_span_labels), self._max_relation_negative_samples)
        )
        neg_span_pairs = [ns[0] for ns in negative_samples]
        neg_span_labels = [ns[1] for ns in negative_samples]
        neg_span_masks = [ns[2] for ns in negative_samples]

        relation_spans = pos_span_pairs + neg_span_pairs
        relation_labels = pos_span_labels + neg_span_labels
        relation_masks = pos_span_masks + neg_span_masks

        if relation_spans:
            rels_sample_masks = np.ones(len(relation_spans))
        else:
            rels_sample_masks = np.zeros(1)

        instance_fields["rels_sample_masks"] = ArrayField(rels_sample_masks, dtype=np.bool)

        instance_fields["relation_masks"] = ListField(
            [ArrayField(np.array(si, dtype=np.int), dtype=np.int) for si in relation_masks])

        instance_fields["rel_span_indices"] = ListField(
            [ArrayField(np.array(si, dtype=np.int), dtype=np.int) for si in relation_spans])

        instance_fields["rel_labels"] = ListField(
            [MultiLabelField(rel_label, label_namespace="rel_labels") for rel_label in relation_labels])

        metadata = {"words": words, "relations": relations}
        instance_fields["metadata"] = MetadataField(metadata)

        return Instance(instance_fields)

    def _too_long(self, span):
        return span[1] - span[0] + 1 > self._max_span_width


if __name__ == "__main__":
    reader = SpartReader(max_span_width=10)

    dataset = list(reader.read("https://fh-aachen.sciebo.de/s/MjcrDC3gDjwU7Vd/download"))

    print(f"read {len(dataset)} samples")
