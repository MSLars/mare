from typing import Dict, List
from allennlp.common import JsonDict
from allennlp.data import Instance, Field, Token
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import TextField, MetadataField, SpanField, ListField
from allennlp.predictors import Predictor


@Predictor.register("span_predictor")
class SpanBasedPredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = [Token(t) for t in json_dict["tokens"]]
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
        return output_dict["relations"]

    #@timeit
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        output = super().predict_batch_json(inputs)
        return output
