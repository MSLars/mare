from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict

import torch

from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
    TypedStringSpan,
)

from mare.label.extraction import transform_tags_to_relation

TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]


@Metric.register("span_metric")
class SpanRelationMetric(Metric):
    def __init__(
            self,
            ignore_classes: List[str] = None,
    ) -> None:

        self._ignore_classes: List[str] = ignore_classes or []

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(
            self,
            predicted_batches,
            gold_batches
    ):

        for i, predicted_relations in enumerate(predicted_batches):

            gold_relations = set(gold_batches[i])

            for p_rel in predicted_relations:
                if p_rel in gold_relations:
                    self._true_positives[p_rel.label] += 1
                    gold_relations.remove(p_rel)
                else:
                    self._false_positives[p_rel.label] += 1

            for g_rel in gold_relations:
                self._false_negatives[g_rel.label] += 1
            pass

    def get_metric(self, reset: bool = False):
        """
        # Returns
        `Dict[str, float]`
            A Dict per label containing following the span based metrics:
            - precision : `float`
            - recall : `float`
            - f1-measure : `float`
            Additionally, an `overall` key is included, which provides the precision,
            recall and f1-measure for all spans.
        """
        if is_distributed():
            raise RuntimeError(
                "Distributed aggregation for SpanBasedF1Measure is currently not supported."
            )
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        # for tag in all_tags:
        #     precision, recall, f1_measure = self._compute_metrics(
        #         self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
        #     )
        #     precision_key = "precision" + "-" + tag
        #     recall_key = "recall" + "-" + tag
        #     f1_key = "FR-" + "-" + tag
        #     all_metrics[precision_key] = precision
        #     all_metrics[recall_key] = recall
        #     all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )

        all_metrics["PR-overall"] = precision
        all_metrics["RR-overall"] = recall
        all_metrics["FR-overall"] = f1_measure
        if reset:
            self.reset()
        return precision, recall, f1_measure

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

@Metric.register("smart_metric")
class RelationMetric(Metric):
    """
    The Conll SRL metrics are based on exact span matching. This metric
    implements span-based precision and recall metrics for a BIO tagging
    scheme. It will produce precision, recall and F1 measures per tag, as
    well as overall statistics. Note that the implementation of this metric
    is not exactly the same as the perl script used to evaluate the CONLL 2005
    data - particularly, it does not consider continuations or reference spans
    as constituents of the original span. However, it is a close proxy, which
    can be helpful for judging model performance during training. This metric
    works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "I", "O" if using the "BIO" label encoding).
    """

    def __init__(
            self,
            vocabulary: Vocabulary,
            tag_namespace: str = "tags",
            ignore_classes: List[str] = None,
            label_encoding: Optional[str] = "BIO",
            has_mode: bool = False,
            tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None,
            max_relation_width:int = 11,
    ) -> None:
        """
        # Parameters
        vocabulary : `Vocabulary`, required.
            A vocabulary containing the tag namespace.
        tag_namespace : `str`, required.
            This metric assumes that a BIO format is used in which the
            labels are of the format: ["B-LABEL", "I-LABEL"].
        ignore_classes : `List[str]`, optional.
            Span labels which will be ignored when computing span metrics.
            A "span label" is the part that comes after the BIO label, so it
            would be "ARG1" for the tag "B-ARG1". For example by passing:
             `ignore_classes=["V"]`
            the following sequence would not consider the "V" span at index (2, 3)
            when computing the precision, recall and F1 metrics.
            ["O", "O", "B-V", "I-V", "B-ARG1", "I-ARG1"]
            This is helpful for instance, to avoid computing metrics for "V"
            spans in a BIO tagging scheme which are typically not included.
        label_encoding : `str`, optional (default = `"BIO"`)
            The encoding used to specify label span endpoints in the sequence.
            Valid options are "BIO", "IOB1", "BIOUL" or "BMES".
        tags_to_spans_function : `Callable`, optional (default = `None`)
            If `label_encoding` is `None`, `tags_to_spans_function` will be
            used to generate spans.
        """
        if label_encoding and tags_to_spans_function:
            raise ConfigurationError(
                "Both label_encoding and tags_to_spans_function are provided. "
                'Set "label_encoding=None" explicitly to enable tags_to_spans_function.'
            )
        if label_encoding:
            if label_encoding not in ["BIO", "IOB1", "BIOUL", "BMES"]:
                raise ConfigurationError(
                    "Unknown label encoding - expected 'BIO', 'IOB1', 'BIOUL', 'BMES'."
                )
        elif tags_to_spans_function is None:
            raise ConfigurationError(
                "At least one of the (label_encoding, tags_to_spans_function) should be provided."
            )

        self._label_encoding = label_encoding
        self._tags_to_spans_function = tags_to_spans_function
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._ignore_classes: List[str] = ignore_classes or []

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

        self.vocab = vocabulary

        self.has_mode = has_mode

        self.max_relation_width = max_relation_width

    def __call__(
            self,
            tags: torch.Tensor,
            mask: Optional[torch.BoolTensor] = None,
            correct_rels: Optional= None,
    ):
        """
        # Parameters
        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        prediction_map : `torch.Tensor`, optional (default = `None`).
            A tensor of size (batch_size, num_classes) which provides a mapping from the index of predictions
            to the indices of the label vocabulary. If provided, the output label at each timestep will be
            `vocabulary.get_index_to_token_vocabulary(prediction_map[batch, argmax(predictions[batch, t]))`,
            rather than simply `vocabulary.get_index_to_token_vocabulary(argmax(predictions[batch, t]))`.
            This is useful in cases where each Instance in the dataset is associated with a different possible
            subset of labels from a large label-space (IE FrameNet, where each frame has a different set of
            possible roles associated with it).
        """
        if mask is None:
            mask = torch.ones_like(tags).bool()

        prediction_map = None

        tags, mask = self.detach_tensors(
            tags, mask
        )
        if isinstance(tags, torch.Tensor):
            tags = tags.tolist()

        bio_tags = [[self._label_vocabulary[t] for t in batch] for batch in tags]

        # Iterate over timesteps in batch.
        # batch_size = gold_labels.size(0)
        # TODO Weiche Entitätsgrenzen erlauben (Wie auch im original Paper)
        # zip bio tags and correct rels
        for i, predicted_tags in enumerate(bio_tags):
            predicted_relations = transform_tags_to_relation(predicted_tags, max_inner_range= self.max_relation_width, has_mode=self.has_mode)

            gold_relations = set(correct_rels[i])

            for p_rel in predicted_relations:
                if p_rel in gold_relations:
                    self._true_positives[p_rel.label] += 1
                    gold_relations.remove(p_rel)
                else:
                    self._false_positives[p_rel.label] += 1

            for g_rel in gold_relations:
                self._false_negatives[g_rel.label] += 1
            pass

    @staticmethod
    def _handle_continued_spans(spans: List[TypedStringSpan]) -> List[TypedStringSpan]:
        """
        The official CONLL 2012 evaluation script for SRL treats continued spans (i.e spans which
        have a `C-` prepended to another valid tag) as part of the span that they are continuing.
        This is basically a massive hack to allow SRL models which produce a linear sequence of
        predictions to do something close to structured prediction. However, this means that to
        compute the metric, these continuation spans need to be merged into the span to which
        they refer. The way this is done is to simply consider the span for the continued argument
        to start at the start index of the first occurrence of the span and end at the end index
        of the last occurrence of the span. Handling this is important, because predicting continued
        spans is difficult and typically will effect overall average F1 score by ~ 2 points.
        # Parameters
        spans : `List[TypedStringSpan]`, required.
            A list of (label, (start, end)) spans.
        # Returns
        A `List[TypedStringSpan]` with continued arguments replaced with a single span.
        """
        span_set: Set[TypedStringSpan] = set(spans)
        continued_labels: List[str] = [
            label[2:] for (label, span) in span_set if label.startswith("C-")
        ]
        for label in continued_labels:
            continued_spans = {span for span in span_set if label in span[0]}

            span_start = min(span[1][0] for span in continued_spans)
            span_end = max(span[1][1] for span in continued_spans)
            replacement_span: TypedStringSpan = (label, (span_start, span_end))

            span_set.difference_update(continued_spans)
            span_set.add(replacement_span)

        return list(span_set)

    def get_metric(self, reset: bool = False):
        """
        # Returns
        `Dict[str, float]`
            A Dict per label containing following the span based metrics:
            - precision : `float`
            - recall : `float`
            - f1-measure : `float`
            Additionally, an `overall` key is included, which provides the precision,
            recall and f1-measure for all spans.
        """
        if is_distributed():
            raise RuntimeError(
                "Distributed aggregation for SpanBasedF1Measure is currently not supported."
            )
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(
                self._true_positives[tag], self._false_positives[tag], self._false_negatives[tag]
            )
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "FR-" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(
            sum(self._true_positives.values()),
            sum(self._false_positives.values()),
            sum(self._false_negatives.values()),
        )

        all_metrics["PR-overall"] = precision
        all_metrics["RR-overall"] = recall
        all_metrics["FR-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = true_positives / (true_positives + false_positives + 1e-13)
        recall = true_positives / (true_positives + false_negatives + 1e-13)
        f1_measure = 2.0 * (precision * recall) / (precision + recall + 1e-13)
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)