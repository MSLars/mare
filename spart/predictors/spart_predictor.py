import json
from typing import Dict, List

import numpy as np
from allennlp.common import JsonDict
from allennlp.common.file_utils import cached_path
from allennlp.data import Field, Instance, Token
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import ArrayField, ListField, MetadataField, SpanField, TextField
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from spart.data.dataset_readers.spart_reader import create_mask


@Predictor.register("spart")
class SpartPredictor(Predictor):
  def _json_to_instance(self, json_dict: JsonDict) -> Instance:
    if "text" in json_dict:
      text = json_dict["text"]
      words = [text[t["span"]["start"]: t["span"]["end"]] for t in json_dict["tokens"]]
    else:
      words = json_dict["tokens"]

    tokens = [Token(w) for w in words]
    # Attribut (_dataset_reader._token_indexers ) wird durch unseren DataReader hinzugefügt!
    # Nicht allgemein gültig...
    token_indexers = self._dataset_reader._token_indexers
    sequence = TextField(tokens, token_indexers=token_indexers)

    context_size = len(words) + 1
    spans = []
    span_masks = []
    for start, end in enumerate_spans(tokens, max_span_width= self._dataset_reader._max_span_width):
      spans.append(SpanField(start, end, sequence))
      span_masks.append(create_mask(start, end, context_size))

    span_field = ListField(spans)
    # span_tuples = [(span.span_start, span.span_end) for span in spans]
    span_mask_field = ListField([ArrayField(np.array(si, dtype=np.int), dtype=np.int) for si in span_masks])
    instance_fields: Dict[str, Field] = {"tokens": sequence,
                                         "metadata": MetadataField({"words": [x.text for x in tokens]}),
                                         "spans": span_field,
                                         "span_masks": span_mask_field}
    return Instance(instance_fields)

  def predict_json(self, inputs: JsonDict) -> JsonDict:
    output_dict = super().predict_json(inputs)
    return output_dict["relations"]

  # @timeit
  def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
    output = super().predict_batch_json(inputs)
    return output


if __name__ == "__main__":

  import spart.data.dataset_readers.spart_reader
  import spart.models.spart

  model_archive = load_archive("/home/lars/Projects/allen_spert/models/test/model.tar.gz")

  predictor = SpartPredictor(model_archive.model, model_archive.dataset_reader)

  file_path = cached_path("https://fh-aachen.sciebo.de/s/3GpXCZLhjwm2SJU/download")

  with open(file_path, "r") as f:
    lines = f.readlines()

    for line in lines:
      doc_text = json.loads(line)

      res = predictor.predict_json(doc_text)

      i=1