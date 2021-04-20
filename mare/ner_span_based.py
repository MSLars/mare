import logging
from typing import Any, Dict, List, Optional, Callable

import torch
from allennlp.modules.span_extractors import SpanExtractor
from torch.nn import functional as F
from overrides import overrides

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, FeedForward, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from mare.label.extraction import Span, transform_spans_to_relation
from mare.metric import RelationMetric, SpanRelationMetric
from mare.ner_metric import NERMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("span_ner_tagger")
class NERTagger(Model):
    """
    Named entity recognition module of DyGIE model.

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 span_extractor: SpanExtractor,
                 feedforward: FeedForward,
                 ner_threshold: float = 0.65,
                 max_inner_range: float = 18,
                 metadata: List[Dict[str, Any]] = None,
                 label_namespace: str = "ner_labels",
                 regularizer: Optional[RegularizerApplicator] = None,
                 initializer: InitializerApplicator = InitializerApplicator()) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        self._include_trigger = False
        for label in vocab.get_token_to_index_vocabulary(label_namespace):
            if "trigger" in label:
                self._include_trigger = True

        self.label_namespace = label_namespace
        self._n_labels = self.vocab.get_vocab_size(label_namespace)

        # null_label = vocab.get_token_index("", label_namespace)
        # assert null_label == 0

        self._ner_threshold = ner_threshold
        self._max_inner_range = max_inner_range
        self._ner_scorer = torch.nn.ModuleDict()

        self._text_field_embedder = text_field_embedder

        self._span_extractor = span_extractor

        self._ner_scorer = torch.nn.Sequential(
            TimeDistributed(feedforward),
            TimeDistributed(torch.nn.Linear(
                feedforward.get_output_dim(),
                self._n_labels)))

        self._relation_f1_metric = RelationMetric(
            vocab, tag_namespace=label_namespace,
        )

        self._ner_metric = NERMetrics(self._n_labels)
        self._relation_metric = SpanRelationMetric()

        self._loss = torch.nn.BCEWithLogitsLoss(reduction="sum")

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                tokens: TextFieldTensors,
                spans: torch.IntTensor,
                ner_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        mask = util.get_text_field_mask(tokens)
        span_mask = (spans[:, :, 0] >= 0)
        sentence_lengths = mask.sum(dim=1).long()

        embedded_text_input = self._text_field_embedder(tokens)
        span_embeddings = self._span_extractor(embedded_text_input, spans, span_mask)

        # cls_h = embedded_text_input[:, 0, :].unsqueeze(1).repeat(1, span_embeddings.shape[1], 1)

        # span_vectors = torch.cat((span_embeddings, cls_h), dim=2)

        span_vectors = span_embeddings

        # 32, 90, 1000
        ner_scores = self._ner_scorer(span_vectors)
        # Give large negative scores to masked-out elements.
        mask = span_mask.unsqueeze(-1)
        ner_scores = util.replace_masked_values(ner_scores, mask.bool(), -1e20)
        # The dummy_scores are the score for the null label.
        # dummy_dims = [ner_scores.size(0), ner_scores.size(1), 1]
        # dummy_scores = ner_scores.new_zeros(*dummy_dims)
        # ner_scores_logits = torch.cat((dummy_scores, ner_scores), -1)

        ner_scores_probs = torch.sigmoid(ner_scores)

        # predictions = self.predict(ner_scores.detach().cpu(),
        #                            spans.detach().cpu(),
        #                            span_mask.detach().cpu(),
        #                            metadata)
        #
        # output_dict = {"predictions": predictions}

        relations = self.extract_relations(spans, ner_scores_probs)

        output_dict = {"relations": relations}

        if ner_labels is not None:
            self._ner_metric(ner_scores_probs, ner_labels, span_mask)

            self._relation_metric(relations, [m["relations"] for m in metadata])

            loss = self._loss(ner_scores, ner_labels.float())

            output_dict["loss"] = loss

        return output_dict

    def extract_relations(self, spans, predicted_ner):
        num_batches = spans.shape[0]
        result = []
        for batch in range(num_batches):
            span_containing_ent = (predicted_ner[batch] > self._ner_threshold).sum(axis=1)
            entity_span_idx = torch.nonzero(span_containing_ent)
            if len(entity_span_idx) == 0:
                result += [[]]
                continue
            entity_labels = predicted_ner[batch][entity_span_idx].squeeze(1).argmax(axis=1).tolist()
            entity_spans = spans[batch][entity_span_idx].squeeze(1).tolist()

            batch_spans = []

            for span_tuple, label_idx in zip(entity_spans, entity_labels):
                label = self.vocab.get_token_from_index(label_idx, namespace=self.label_namespace)

                batch_spans += [(label, (span_tuple[0], span_tuple[1]))]

            result += [
                transform_spans_to_relation(batch_spans, max_inner_range=self._max_inner_range, allow_overlaps=True,
                                            include_trigger=self._include_trigger)]

        return result

    # def predict(self, ner_scores, spans, span_mask, metadata):
    #     # TODO(dwadden) Make sure the iteration works in documents with a single sentence.
    #     # Zipping up and iterating iterates over the zeroth dimension of each tensor; this
    #     # corresponds to iterating over sentences.
    #     predictions = []
    #     zipped = zip(ner_scores, spans, span_mask, metadata)
    #     for ner_scores_sent, spans_sent, span_mask_sent, sentence in zipped:
    #         predicted_scores_raw, predicted_labels = ner_scores_sent.max(dim=1)
    #         softmax_scores = F.softmax(ner_scores_sent, dim=1)
    #         predicted_scores_softmax, _ = softmax_scores.max(dim=1)
    #         ix = (predicted_labels != 0) & span_mask_sent.bool()
    #
    #         predictions_sent = []
    #         zip_pred = zip(predicted_labels[ix], predicted_scores_raw[ix],
    #                        predicted_scores_softmax[ix], spans_sent[ix])
    #         for label, label_score_raw, label_score_softmax, label_span in zip_pred:
    #             label_str = self.vocab.get_token_from_index(label.item(), self._active_namespace)
    #             span_start, span_end = label_span.tolist()
    #             ner = [span_start, span_end, label_str, label_score_raw.item(),
    #                    label_score_softmax.item()]
    #             prediction = document.PredictedNER(ner, sentence, sentence_offsets=True)
    #             predictions_sent.append(prediction)
    #
    #         predictions.append(predictions_sent)
    #
    #     return predictions

    # TODO(dwadden) This code is repeated elsewhere. Refactor.
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        precision, recall, f1 = self._ner_metric.get_metric(reset)
        prefix = "span_ner_"
        to_update = {f"{prefix}_precision": precision,
                     f"{prefix}_recall": recall,
                     f"{prefix}_f1": f1}
        res.update(to_update)

        precision, recall, f1 = self._relation_metric.get_metric(reset)
        prefix = "relation"
        to_update = {f"{prefix}_precision": precision,
                     f"{prefix}_recall": recall,
                     f"{prefix}_f1": f1}
        res.update(to_update)

        return res
