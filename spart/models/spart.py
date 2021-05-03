from allennlp.models.model import Model
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import TextFieldEmbedder
from torch import nn as nn
from typing import List, Dict, Any, Tuple
from spart.data.dataset_readers import spart_reader
from spart.data.dataset_readers.spart_reader import relation_args_names
from spart.metrics.fbeta_measure import FBetaMeasure
from spart.metrics.fbeta_multi import FBetaMultiLabelMeasure

import torch


def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor


@Model.register("spart")
class Spart(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 hidden_size: int = 768,
                 size_embedding: int = 25,
                 dropout: float = 0.1,
                 rel_filter_threshold:float = 0.5,
                 max_pairs: int = 1000) -> None:
        super(Spart, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder

        self._rel_filter_threshold = rel_filter_threshold
        self._relation_types = vocab.get_vocab_size("rel_labels")
        self._entity_types = vocab.get_vocab_size("ner_labels")
        self._cls_token = 2  # TODO nur für https://huggingface.co/german-nlp-group/electra-base-german-uncased/blob/main/vocab.txt
        self._max_pairs = max_pairs
        self._bert = self._text_field_embedder.token_embedder_tokens._modules["_matched_embedder"].transformer_model

        self.rel_classifier = nn.Linear(hidden_size * 3 + size_embedding * 2, self._relation_types)
        self.entity_classifier = nn.Linear(hidden_size * 2 + size_embedding, self._entity_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        self.dropout = nn.Dropout(dropout)

        self._rel_loss = nn.BCEWithLogitsLoss(reduction='none')
        self._ents_loss = nn.CrossEntropyLoss(reduction='none')  # TODO BCEWithLogitsLoss

        ner_labels = list(vocab.get_index_to_token_vocabulary("ner_labels"))
        ner_labels.remove(0)

        rel_labels = list(vocab.get_index_to_token_vocabulary("rel_labels"))
        rel_labels.remove(0)
        self._f1_relation = FBetaMultiLabelMeasure(average="micro", threshold=self._rel_filter_threshold)
        self._f1_entities = FBetaMeasure(average="micro", labels=ner_labels)

    def forward(self,  # type: ignore
                tokens: TextFieldTensors,
                spans: torch.IntTensor,
                ner_labels: torch.IntTensor = None,
                rel_span_indices: torch.IntTensor = None,
                rel_labels: torch.IntTensor = None,
                span_masks: torch.IntTensor = None,
                relation_masks: torch.IntTensor = None,
                rels_sample_masks: torch.BoolTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Dict]:
        embedded_text_input = self._text_field_embedder(tokens)
        try:
            entity_ctx = self._bert(tokens["tokens"]["token_ids"]).last_hidden_state[:, 0, :]
        except AttributeError:
            entity_ctx = self._bert(tokens["tokens"]["token_ids"])[0][:, 0, :]
        embedded_text_input = torch.cat((entity_ctx.unsqueeze(1), embedded_text_input), dim=1)
        batch_size = embedded_text_input.shape[0]

        entity_sizes = spans[:, :, 1] - spans[:, :, 0] + 1

        size_embeddings = self.size_embeddings(entity_sizes)

        entity_clf, entity_spans_pool = self._classify_entities(embedded_text_input,
                                                                span_masks, size_embeddings, entity_ctx)

        # TODO If we have no gold entities, we cannot specify relation candidates!
        # entity_max_logits_index = entity_clf.max(dim=2).indices
        # relation_candidates = []
        # relation_masks = []
        # for batch in range(entity_max_logits_index.shape[0]):
        #
        #     entity_indices = entity_max_logits_index[batch].nonzero(as_tuple=True)[0]
        #
        #     new_candidates = list(itertools.permutations(entity_indices.tolist(), 2))
        #
        #     for nc in new_candidates:
        #
        #         start_entity_span = tuple(spans[batch][nc[0]].tolist())
        #         end_entity_span = tuple(spans[batch][nc[1]].tolist())
        #
        #         relation_masks += [create_rel_mask(start_entity_span, end_entity_span, embedded_text_input.shape[1])]
        #
        #     relation_candidates += []

        #TODO  wir haben zur evaluation KEINE Label, die zu diesen labeln passen, wir müssen die von Span Labeling usw. wieder nutzen!
        # rel_span_indices = torch.tensor(relation_candidates, device=entity_clf.device)

        # classify relations
        if rel_labels is None:
            ctx_size = embedded_text_input.shape[1]

            entity_sample_masks = torch.ones((batch_size, entity_clf.shape[1]))

            rel_span_indices, relation_masks, rel_sample_masks = self._filter_spans(entity_clf, spans, entity_sample_masks, ctx_size)

            rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
            h_large = embedded_text_input.unsqueeze(1).repeat(1, max(min(rel_span_indices.shape[1], self._max_pairs), 1), 1, 1)
            rel_clf = torch.zeros([batch_size, rel_span_indices.shape[1], self._relation_types]).to(
                self.rel_classifier.weight.device)
        else:
            h_large = embedded_text_input.unsqueeze(1).repeat(1, max(min(rel_span_indices.shape[1], self._max_pairs), 1), 1,
                                                              1)
            rel_clf = torch.zeros([batch_size, rel_span_indices.shape[1], self._relation_types]).to(
                self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, rel_span_indices.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        rel_span_indices, relation_masks, h_large, i)
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        converted_relations = []

        for batch in range(batch_size):
            batch_pred_entities, batch_pred_relations = self.convert_predictions(entity_clf[batch].unsqueeze(0),
                                                                                 rel_clf[batch].unsqueeze(0),
                                                                                 rel_span_indices[batch].unsqueeze(0),
                                                                                 spans[batch].unsqueeze(0),
                                                                                 entity_sample_masks[batch].unsqueeze(0),
                                                                                 self._rel_filter_threshold,)

            batch_converted_relations = []
            for pred_relation in batch_pred_relations[0]:

                h_name, t_name = sorted(relation_args_names[pred_relation[2]])
                converted_relation = {
                    "name": pred_relation[2],
                    "ents":[
                        {
                            "name":h_name,
                            "start":pred_relation[0][0],
                            "end":pred_relation[0][1],
                        },
                        {
                            "name":t_name,
                            "start":pred_relation[1][0],
                            "end":pred_relation[1][1],
                        },
                    ]
                }

                batch_converted_relations += [converted_relation]

            converted_relations += [batch_converted_relations]

        if ner_labels and rel_labels:

            batch_loss = self.compute_loss(entity_logits=entity_clf, rel_logits=rel_clf, rel_types=rel_labels,
                                           entity_types=ner_labels,
                                           rel_sample_masks=rels_sample_masks)

            self._f1_entities(entity_clf, ner_labels)
            #self._f1_relation(rel_clf, rel_labels.bool())
            self._f1_relation(rel_clf.squeeze(), rel_labels.bool().squeeze())

            return {"loss": batch_loss}

        return {"relations": converted_relations}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        ent_metrics = self._f1_entities.get_metric(reset=reset)
        rel_metrics = self._f1_relation.get_metric(reset=reset)
        return {
            "ner_p": ent_metrics["precision"],
            "ner_r": ent_metrics["recall"],
            "ner_f": ent_metrics["fscore"],
            "rel_p": rel_metrics["precision"],
            "rel_r": rel_metrics["recall"],
            "rel_f": rel_metrics["fscore"],
        }

    def convert_predictions(self, batch_entity_clf: torch.tensor, batch_rel_clf: torch.tensor,
                            batch_rels: torch.tensor, spans, span_masks, rel_filter_threshold: float,
                            no_overlapping: bool = False):
        # get maximum activation (index of predicted entity type)
        batch_entity_types = batch_entity_clf.argmax(dim=-1)
        # apply entity sample mask
        batch_entity_types *= span_masks.long()

        # apply threshold to relations
        batch_rel_clf[batch_rel_clf < rel_filter_threshold] = 0

        batch_pred_entities = []
        batch_pred_relations = []

        for i in range(batch_rel_clf.shape[0]):
            # get model predictions for sample
            entity_types = batch_entity_types[i]
            entity_spans = spans[i]
            entity_clf = batch_entity_clf[i]
            rel_clf = batch_rel_clf[i]
            rels = batch_rels[i]

            # convert predicted entities
            sample_pred_entities = self._convert_pred_entities(entity_types, entity_spans,
                                                          entity_clf)

            # convert predicted relations
            sample_pred_relations = self._convert_pred_relations(rel_clf, rels,
                                                            entity_types, entity_spans)

            # if no_overlapping:
            #     sample_pred_entities, sample_pred_relations = remove_overlapping(sample_pred_entities,
            #                                                                      sample_pred_relations)

            batch_pred_entities.append(sample_pred_entities)
            batch_pred_relations.append(sample_pred_relations)

        return batch_pred_entities, batch_pred_relations

    def compute_loss(self, entity_logits, rel_logits, rel_sample_masks, entity_types, rel_types):
        entity_sample_masks = entity_types != -1
        entity_types[entity_types == -1] = 0
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._ents_loss(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # relation loss
        #rel_sample_masks = rel_types != -1
        #rel_types[rel_types == -1] = 0
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_loss(rel_logits, rel_types.float())
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            # joint loss
            return entity_loss + rel_loss
        else:
            # corner case: no positive/negative relation samples
            return entity_loss

    def _classify_entities(self, h, entity_masks, size_embeddings, entity_ctx):
        # max pool entity candidate spans
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool


    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]

        # create chunks if necessary
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # relation context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx = rel_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # create relation candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # classify relation candidates
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(torch.tensor(spart_reader.create_rel_mask(s1, s2, ctx_size)))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = padded_stack(batch_relations).to(device)
        batch_rel_masks = padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def _convert_pred_entities(self, entity_types: torch.tensor, entity_spans: torch.tensor,
                               entity_scores: torch.tensor):

        # get entities that are not classified as 'None'
        valid_entity_indices = entity_types.nonzero().view(-1)

        if not valid_entity_indices.tolist():
            return []

        pred_entity_types = entity_types[valid_entity_indices]
        pred_entity_spans = entity_spans[valid_entity_indices]
        pred_entity_scores = torch.gather(entity_scores[valid_entity_indices], 1,
                                          pred_entity_types.unsqueeze(1)).view(-1)

        # convert to tuples (start, end, type, score)
        converted_preds = []
        for i in range(pred_entity_types.shape[0]):
            label_idx = pred_entity_types[i].item()
            entity_type = self.vocab.get_index_to_token_vocabulary("ner_labels")[label_idx]

            start, end = pred_entity_spans[i].tolist()
            score = pred_entity_scores[i].item()

            converted_pred = (start, end, entity_type, score)
            converted_preds.append(converted_pred)

        return converted_preds

    def _convert_pred_relations(self, rel_clf: torch.tensor, rels: torch.tensor,
                                entity_types: torch.tensor, entity_spans: torch.tensor):
        rel_class_count = rel_clf.shape[1]
        rel_clf = rel_clf.view(-1)

        # get predicted relation labels and corresponding entity pairs
        rel_nonzero = rel_clf.nonzero().view(-1)
        pred_rel_scores = rel_clf[rel_nonzero]

        pred_rel_types = (rel_nonzero % rel_class_count) + 1  # model does not predict None class (+1)
        valid_rel_indices = rel_nonzero // rel_class_count
        valid_rels = rels[valid_rel_indices]

        # get masks of entities in relation
        pred_rel_entity_spans = entity_spans[valid_rels].long()

        # get predicted entity types
        pred_rel_entity_types = torch.zeros([valid_rels.shape[0], 2])
        if valid_rels.shape[0] != 0:
            pred_rel_entity_types = torch.stack([entity_types[valid_rels[j]] for j in range(valid_rels.shape[0])])

        # convert to tuples ((head start, head end, head type), (tail start, tail end, tail type), rel type, score))
        converted_rels = []
        check = set()

        for i in range(pred_rel_types.shape[0]):
            label_idx = pred_rel_types[i].item()
            pred_rel_type = self.vocab.get_index_to_token_vocabulary("rel_labels")[label_idx-1]
            pred_head_type_idx, pred_tail_type_idx = pred_rel_entity_types[i][0].item(), pred_rel_entity_types[i][
                1].item()
            pred_head_type = self.vocab.get_index_to_token_vocabulary("ner_labels")[pred_head_type_idx]
            pred_tail_type = self.vocab.get_index_to_token_vocabulary("ner_labels")[pred_tail_type_idx]
            score = pred_rel_scores[i].item()

            spans = pred_rel_entity_spans[i]
            head_start, head_end = spans[0].tolist()
            tail_start, tail_end = spans[1].tolist()

            converted_rel = ((head_start, head_end, pred_head_type),
                             (tail_start, tail_end, pred_tail_type), pred_rel_type)
            converted_rel = _adjust_rel(converted_rel)

            if converted_rel not in check:
                check.add(converted_rel)
                converted_rels.append(tuple(list(converted_rel) + [score]))

        return converted_rels


def _adjust_rel(rel: Tuple):
    adjusted_rel = rel

    #TODO symmertic relastions
    # if rel[-1].symmetric:
    #     head, tail = rel[:2]
    #     if tail[0] < head[0]:
    #         adjusted_rel = tail, head, rel[-1]

    return adjusted_rel
