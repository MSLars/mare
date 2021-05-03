from pathlib import Path
from typing import Union
from allennlp.models import load_archive
from dygie.predictors import dygie

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


class MockModel:

    def predict_batch_json(self, batch):
        raise NotImplemented()


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

            result += [{"relations": relations}]

        return result