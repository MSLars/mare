from typing import Dict, List

from allennlp.common import JsonDict
from allennlp.data import Instance, Token, Field
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import TextField, SpanField, ListField, MetadataField
from allennlp.predictors import Predictor


@Predictor.register("re_predictor")
class SpanBasedPredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = [Token(t["text"]) for t in json_dict["tokens"]]
        # Attribut (_dataset_reader._token_indexers ) wird durch unseren DataReader hinzugefügt!
        # Nicht allgemein gültig...
        token_indexers = self._dataset_reader._token_indexers
        sequence = TextField(tokens, token_indexers=token_indexers)

        spans = []
        for start, end in enumerate_spans(tokens, max_span_width=10):
            spans.append(SpanField(start, end, sequence))

        span_field = ListField(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]

        instance_fields: Dict[str, Field] = {"tokens": sequence,
                                             "metadata": MetadataField({"words": [x.text for x in tokens]}),
                                             "spans": span_field}
        return Instance(instance_fields)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        output_dict = super().predict_json(inputs)
        id = inputs["id"]
        tokens = inputs["tokens"]
        target_relations = []
        for rel in output_dict["relations"]:
            try:
                head_ent = [e for e in rel["ents"] if e["name"] == "head"][0]
                tail_ent = [e for e in rel["ents"] if e["name"] == "tail"][0]

                head_char_start = tokens[head_ent["start"]]["start"]
                head_char_end = tokens[head_ent["end"]]["stop"]

                tail_char_start = tokens[tail_ent["start"]]["start"]
                tail_char_end = tokens[tail_ent["end"]]["stop"]

                target_relations.append({
                    "head_entity": {
                        "start": head_char_start,
                        "stop": head_char_end
                    },
                    "tail_entity": {
                        "start": tail_char_start,
                        "stop": tail_char_end
                    },
                    "type": rel["name"]
                })
            except:
                continue

        return {"id": id, "relations": target_relations}

    #@timeit
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        output = super().predict_batch_json(inputs)
        return output