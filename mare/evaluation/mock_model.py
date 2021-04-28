import json
from pathlib import Path
from typing import List, Dict, Union

import torch
from allennlp.models import load_archive
from dygie.predictors import dygie
from spert.input_reader import JsonInputReader
from spert.models import SpERT
from spert import prediction, util
from transformers import BertConfig, BertTokenizer
import dygie.data.dataset_readers.dygie


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


class MockModel:

    def predict_batch_json(self, batch):
        raise NotImplemented()


class SpertMockModel(MockModel):
    def __init__(self, model_path):
        types_path = model_path + "/types.json"
        self.config = BertConfig.from_pretrained(model_path, cache_dir=None)

        with open(types_path, "r") as file:
            self.type_defs = json.load(file)

        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        # We need the input reader to create the mappings entity_label_id -> entity_label (relations vice versa)
        self.input_reader = JsonInputReader(types_path, self.tokenizer)

        cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')

        self.rel_threshold = 0.85

        self.model = SpERT.from_pretrained(model_path,
                                      config=self.config,
                                      cls_token=cls_id,
                                      relation_types=15,
                                      entity_types=17,
                                      max_pairs=1000,
                                      prop_drop=0.0,
                                      size_embedding=25,
                                      freeze_transformer=False)

    def convert_to_model_input(self, tokenization: List[str] = None):
        from spert.entities import Token
        doc_tokens = []
        t_id = 0
        # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
        doc_encoding = [self.tokenizer.convert_tokens_to_ids('[CLS]')]

        encoding_to_token_idx = [0]

        # parse tokens
        for i, token_phrase in enumerate(tokenization):

            token_encoding = self.tokenizer.encode(token_phrase, add_special_tokens=False)

            if len(doc_encoding) + len(token_encoding) > 512:
                break

            span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

            token = Token(t_id, i, span_start, span_end, token_phrase)

            encoding_to_token_idx += [i] * len(token_encoding)

            doc_tokens.append(token)
            doc_encoding += token_encoding

        encoding_to_token_idx += [len(encoding_to_token_idx)]
        doc_encoding += [self.tokenizer.convert_tokens_to_ids('[SEP]')]

        return doc_tokens, doc_encoding, encoding_to_token_idx

    @staticmethod
    def create_entity_mask(start, end, context_size):
        mask = torch.zeros(context_size, dtype=torch.bool)
        mask[start:end] = 1
        return mask

    @staticmethod
    def create_entity_candidate_data(encodings, tokens, max_span_size=6):
        from spert.entities import TokenSpan
        token_spans = TokenSpan(tokens)
        token_count = len(token_spans)
        context_size = len(encodings)

        # create entity candidates
        entity_spans = []
        entity_masks = []
        entity_sizes = []

        for size in range(1, max_span_size + 1):
            for i in range(0, (token_count - size) + 1):
                span = token_spans[i:i + size].span
                entity_spans.append(span)
                entity_masks.append(SpertMockModel.create_entity_mask(*span, context_size))
                entity_sizes.append(size)

        # create tensors
        # token indices
        _encoding = encodings
        encodings = torch.zeros(context_size, dtype=torch.long)
        encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

        # masking of tokens
        context_masks = torch.zeros(context_size, dtype=torch.bool)
        context_masks[:len(_encoding)] = 1

        # entities
        if entity_masks:
            entity_masks = torch.stack(entity_masks)
            entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
            entity_spans = torch.tensor(entity_spans, dtype=torch.long)

            # tensors to mask entity samples of batch
            # since samples are stacked into batches, "padding" entities possibly must be created
            # these are later masked during evaluation
            entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
        else:
            # corner case handling (no entities)
            entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
            entity_sizes = torch.zeros([1], dtype=torch.long)
            entity_spans = torch.zeros([1, 2], dtype=torch.long)
            entity_sample_masks = torch.zeros([1], dtype=torch.bool)

        return dict(encodings=encodings.unsqueeze(0),
                    context_masks=context_masks.unsqueeze(0),
                    entity_masks=entity_masks.unsqueeze(0),
                    entity_sizes=entity_sizes.unsqueeze(0),
                    entity_spans=entity_spans.unsqueeze(0),
                    entity_sample_masks=entity_sample_masks.unsqueeze(0))

    def transform_predictions_to_relations(self, predictions, tokens):
        relations = []

        # convert entities
        converted_entities = []
        for entity in predictions[0][0]:
            entity_span = entity[:2]
            span_tokens = util.get_span_tokens(tokens, entity_span)
            entity_type = entity[2].identifier
            converted_entity = dict(type=entity_type, start=span_tokens[0].index, end=span_tokens[-1].index + 1)
            converted_entities.append(converted_entity)
        converted_entities = sorted(converted_entities, key=lambda e: e['start'])


        for rel_cand in predictions[1][0]:

            head, tail = rel_cand[:2]
            head_span, head_type = head[:2], head[2].identifier
            tail_span, tail_type = tail[:2], tail[2].identifier
            head_span_tokens = util.get_span_tokens(tokens, head_span)
            tail_span_tokens = util.get_span_tokens(tokens, tail_span)
            relation_type = rel_cand[2].identifier

            converted_head = dict(type=head_type, start=head_span_tokens[0].index,
                                  end=head_span_tokens[-1].index + 1)
            converted_tail = dict(type=tail_type, start=tail_span_tokens[0].index,
                                  end=tail_span_tokens[-1].index + 1)

            head_idx = converted_entities.index(converted_head)
            tail_idx = converted_entities.index(converted_tail)

            h_role, t_role = sorted(relation_args_names[relation_type])
            # converted_relation = dict(type=relation_type, head=head_idx, tail=tail_idx)
            # converted_relations.append(converted_relation)

            # relation_name = rel_cand[2].identifier
            #
            # h_role, t_role = sorted(relation_args_names[relation_name])
            #
            relations += [{
                "name": relation_type,
                "ents": [{
                    "name": h_role,
                    "start": converted_head["start"],
                    "end": converted_head["end"]-1,
                },{
                    "name": t_role,
                    "start": converted_tail["start"],
                    "end": converted_tail["end"]-1,
                },],
            }]

        return relations

    def predict_batch_json(self, batch: List[Dict[str, List[str]]]):
        results = []
        with torch.no_grad():
            self.model.eval()
            for elem in batch:
                tokens = elem["tokens"]
                spert_tokens, encodings, encoding_to_token_idx = self.convert_to_model_input(tokens)
                instance = self.create_entity_candidate_data(encodings, spert_tokens)

                # mapping from span_id to encoding_span
                # this borders relate to the bert subword encodings
                entity_encoding_spans = instance["entity_spans"].squeeze(0).tolist()

                # result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                #                entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                #                entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                #                inference=True)

                entity_clf, rel_clf, related_spans = self.model(encodings=instance['encodings'],
                                                                context_masks=instance['context_masks'],
                                                                entity_masks=instance['entity_masks'],
                                                                entity_sizes=instance['entity_sizes'],
                                                                entity_spans=instance['entity_spans'],
                                                                entity_sample_masks=instance['entity_sample_masks'],
                                                                inference=True)

                predictions = prediction.convert_predictions(entity_clf, rel_clf, related_spans,
                                                             instance, self.rel_threshold,
                                                             self.input_reader)

                relations = self.transform_predictions_to_relations(predictions,spert_tokens)
                results += [{"relations": relations}]

                # # related_spans contains contains a list of all span indizes that form a relation.
                # # For each element in related_spans rel_clf contains the classification logits.
                # # we first extract the maximum indices per element from these logits to get the relation label.
                # # aftwerwards we convert both labels and span indices to python lists
                # # the span indices refer to span definitions in instance["entity_spans"]
                #
                # batch_rel_max = rel_clf.max(dim=-1)
                # batch_rel_clf = batch_rel_max[1].squeeze(0)
                # batch_rel_prob = batch_rel_max[0].squeeze(0)
                # relation_span_list = related_spans.squeeze(0).tolist()
                # # apply threshold to relations
                # batch_rel_clf[batch_rel_prob < 0.4] = -1
                # relation_label = batch_rel_clf.tolist()
                #
                # relatoins = []
                #
                # for label, span_idx in zip(relation_label, relation_span_list):
                #     # Receive the name of the relation label
                #     # Spert manages this mapping in a DataReader class
                #     relation_name = self.input_reader.get_relation_type(label + 1).identifier
                #
                #     if relation_name == "None":
                #         continue
                #
                #     if relation_name == "None":
                #         continue
                #
                #     # Get the ordered argument names (not needed)
                #     # h_arg_name, t_arg_name = [n for n in sorted(self.type_defs["relations"][relation_name]["args"])]
                #     h_arg_name, t_arg_name  = sorted(relation_args_names[relation_name])
                #     # Get the spans for all arguments
                #     # These indices refer to the Bert Subword Tokenization
                #     # We need to transform these afterwards
                #     span_encoding = [entity_encoding_spans[s] for s in span_idx]
                #     h_encoding_span = [encoding_to_token_idx[encoding_idx] for encoding_idx in span_encoding[0]]
                #     t_encoding_span = [encoding_to_token_idx[encoding_idx] for encoding_idx in span_encoding[1]]
                #
                #     relation = {"name": relation_name, "ents": [
                #                                                 {"name": h_arg_name,
                #                                                  "start": h_encoding_span[0],
                #                                                  "end": h_encoding_span[1] - 1,
                #                                                  # "tokens": tokens[t_encoding_span[0]: t_encoding_span[1]]
                #                                                  },
                #                                                 {"name": t_arg_name,
                #                                                  "start": t_encoding_span[0],
                #                                                  "end": t_encoding_span[1] - 1,
                #                                                  #"tokens": tokens[t_encoding_span[0]: t_encoding_span[1]]
                #                                                 },
                #     ]}
                #
                #     relatoins += [relation]
                #
                # results += [{"relations": relatoins}]

        return results


class SpertAltMockModel(SpertMockModel):
    def __init__(self, model_path, types_path):
        super().__init__(model_path, types_path)


    def predict_batch_json(self, batch: List[Dict[str, List[str]]]):
        results = super().predict_batch_json(batch)

        print(len(results))


get_trigger_type_for_relation = {
    "Accident": "trigger",
    "CanceledRoute": "trigger",
    "CanceledStop": "trigger",
    "Delay": "trigger",
    "Disaster": "type",
    "Obstruction": "trigger",
    "RailReplacementService": "trigger",
    "TrafficJam": "trigger",
    "Acquisition": "buyer",
    "Insolvency": "company",
    "Layoffs": "company",
    "Merger": "old",
    "OrganizationLeadership": "organization",
    "SpinOff": "parent",
    "Strike": "company"
}


class DygieppMockModel(MockModel):

    def __init__(self, model_path: Union[str, Path]):
        archive = load_archive(model_path)
        archive.model.eval()

        self._predictor = dygie.predictors.DyGIEPredictor.from_archive(archive)
        self._model = archive.model
        print(self._predictor)

    def predict_batch_json(self, batch):

        result = []

        for elem in batch:

            tokens = elem["tokens"]

            instance = {
                "doc_key": "xyz",
                "dataset": "smart_data",
                "sentences": [elem["tokens"]]
            }

            relations = []

            prediction = self._predictor.predict(instance)

            for pred_event in prediction["predicted_events"][0]:
                pred_trigger = pred_event[0]
                relation = {"name": pred_trigger[2], "ents": []}

                # add Trigger to relation
                trigger_type = get_trigger_type_for_relation[pred_trigger[2]]

                relation["ents"] += [{
                    "start": pred_trigger[0],
                    "end": pred_trigger[1],
                    "name": trigger_type,
                    #"tokens": tokens[pred_trigger[0]: pred_trigger[1]+1]
                }]

                for pred_argument in pred_event[1:]:
                    ent = {}
                    ent["start"] = pred_argument[0]
                    ent["end"] = pred_argument[1]
                    ent["name"] = pred_argument[2]
                    #ent["tokens"] = tokens[pred_argument[0]: pred_argument[1]+1]
                    relation["ents"] += [ent]

                relations += [relation]

            result +=  [{"relations": relations}]

        return result


if __name__ == "__main__":
    # model_path = "/home/lars/Projects/spert/data/models/smart_data_15_do_03"
    # types_path = "/home/lars/Projects/spert/data/datasets/smartData/new_format/spert_smart_data_types.json"
    #
    # mock_model = SpertMockModel(model_path, types_path)
    # results = mock_model.predict_batch_json([{"tokens": ["Die", "Fusion", "von", "Schlecker", "e.K.", "mit", "der", "Rossmann", "GmbH",
    #                                "wurde", "offiziell", "bestätigt", "."]}])
    # print(results)

    model_path = "/home/lars/Projects/dygiepp/models/smart_data_important/dygiepp_smart_data.tar.gz"

    mock_model = DygieppMockModel(model_path)

    results = mock_model.predict_batch_json(
        [{"tokens": ["Die", "Fusion", "von", "Schlecker", "e.K.", "mit", "der", "Rossmann", "GmbH",
                                    "wurde", "offiziell", "bestätigt", "."]}])

    print(results)