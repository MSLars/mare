from typing import Dict, List
from allennlp.common import JsonDict
from allennlp.data import Instance, Field, Token
from allennlp.data.fields import TextField, MetadataField
from allennlp.predictors import Predictor

from mare.label.extraction import transform_tags_to_relation
import spacy


@Predictor.register("seq_lab_elmo_pred")
class SequenceLabElmoPredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = [Token(t) for t in json_dict["tokens"]]
        # Attribut (_dataset_reader._token_indexers ) wird durch unseren DataReader hinzugefügt!
        # Nicht allgemein gültig...
        token_indexers = self._dataset_reader._token_indexers
        sequence = TextField(tokens, token_indexers=token_indexers)
        instance_fields: Dict[str, Field] = {"tokens": sequence,
                                             "metadata": MetadataField({"words": [x.text for x in tokens]})}
        return Instance(instance_fields)

    @staticmethod
    def transform_relations_to_format(relations) -> JsonDict:
        result = {"relations": []}
        for rel in relations:
            rel_name = rel.label
            ents = []
            for span in rel.spans:
                start = span.span[0]
                end = span.span[1]
                role = span.role
                ents.append({"name": role, "start": start, "end": end})
            result["relations"].append({"name": rel_name, "ents": ents})

        return result

    def _post_processing_prediction(self, prediction):
        relations = transform_tags_to_relation(prediction["tags"], max_inner_range=11, has_mode=False, include_trigger=False)
        return self.transform_relations_to_format(relations)

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        output_dict = super().predict_json(inputs)
        return self._post_processing_prediction(output_dict)

    #@timeit
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        output = super().predict_batch_json(inputs)
        return [self._post_processing_prediction(entry) for entry in output]


@Predictor.register("seq_lab_elmo_pred_sentence")
class SequenceLabElmoPredictorSentence(SequenceLabElmoPredictor):
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raw_text = json_dict["text"]

        nlp = spacy.load("de_core_news_sm")
        doc = nlp(raw_text)

        tokens = {"tokens": [t.text for t in doc]}
        return super()._json_to_instance(tokens)



