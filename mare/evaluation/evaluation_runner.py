# JSON Structure:
#
# {
#     "relations": [
#         {
#             "name": name,
#             "ents": [
#                 {
#                     "name": role,
#                     "start": start,
#                     "end": end
#                 }
#             ]
#         }
#     ]
# }
#
# Ideen:
# Soft Constrained wie im Paper
# Hard Constrained, das alle Argumente als mandatory ansieht
#
# Auf Relationsebene und nicht auf Dokumentebene testen?
import srsly
import allennlp.common.util as common_util
from allennlp.common.file_utils import cached_path
from sklearn.metrics import f1_score, precision_score, recall_score
import fastavro
import pandas as pd
import json
from allennlp.predictors import Predictor
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import click
from mare.evaluation.evaluation_strategies import *
from allennlp.models.archival import load_archive
from mare.evaluation.timeit_decorator import timeit
import shutil


def jsonl_df(filepath):
    cacched_file_path = cached_path(filepath)
    return list(srsly.read_jsonl(cacched_file_path))


def transform_entities_char_spans_to_token_spans(rel, tokens):
    token_spans = []
    for arg in rel["args"]:
        start = arg["conceptMention"]["span"]["start"]
        end = arg["conceptMention"]["span"]["end"]

        token_start_index = -1
        token_end_index = -1
        for idx, t in enumerate(tokens):
            t_start = t["span"]["start"]
            t_end = t["span"]["end"]
            if t_start == start:
                token_start_index = idx
            if t_end == end:
                token_end_index = idx

        if token_start_index == -1:
            print("Start nicht gefunden")
        if token_end_index == -1:
            print("Ende nicht gefunden")

        token_spans += [{
            "name": arg["role"],
            "start": token_start_index,
            "end": token_end_index
        }]
    return token_spans


def extract_gold(row):
    gold = {"tokens": [row["text"][t["span"]["start"]:t["span"]["end"]] for t in row["tokens"]],
            "relations": []}
    for rel in row["relationMentions"]:
        rel_name = rel["name"]
        ents = transform_entities_char_spans_to_token_spans(rel, row["tokens"])

        gold["relations"] += [{
            "name": rel_name,
            "ents": ents
        }]

    return gold


class EvaluationRunner:
    _file_suffix = ".json"
    GOLD_LABEL = "gold_labels"
    PRED_LABEL = "pred_labels"
    GOLD_DATA = "gold_data"
    PRED_DATA = "pred_data"
    STRATS = "strategies"

    def __init__(self, model, evaluation_path, dir=""):
        self._store = {}
        self._dir = dir
        self.model = model
        self.df = jsonl_df(evaluation_path)

    def _prepare_store(self, run_names):
        self._store[self.GOLD_DATA] = []
        self._store[self.PRED_DATA] = []
        self._store[self.STRATS] = {}
        for name in run_names:
            self._store[self.STRATS][name] = {
                self.GOLD_LABEL: [],
                self.PRED_LABEL: [],
            }

    def evaluate(self, **run_strategies):
        self._prepare_store(run_strategies.keys())

        batch_size = 1

        batch = []
        gold_for_batch = []

        pbar = tqdm(total=len(self.df))
        for row in self.df:
            tokens = EvaluationRunner._prepare_row(row)
            to_predict = {"tokens": tokens}

            gold = extract_gold(row)

            batch.append(to_predict)
            gold_for_batch.append(gold)

            if len(batch) == batch_size:
                self._predict_and_evaluate_on_batch(batch, gold_for_batch, run_strategies)
                batch = []
                gold_for_batch = []
                pbar.update(batch_size)

        if len(batch) > 0:
            self._predict_and_evaluate_on_batch(batch, gold_for_batch, run_strategies)
            pbar.update(len(batch))

    def _predict_and_evaluate_on_batch(self, batch, gold_for_batch, run_strategies):

        predictions = self.model.predict_batch_json(batch)
        for idx, (current_pred, current_gold) in enumerate(zip(predictions, gold_for_batch)):
            self._evaluate_on_single(current_gold, current_pred, run_strategies)

    def _evaluate_on_single(self, current_gold, current_pred, run_strategies):
        self._store[self.GOLD_DATA].append(current_gold)
        self._store[self.PRED_DATA].append(current_pred)
        for run_name, strategy in run_strategies.items():
            self._perform_strategy(current_gold, current_pred, run_name, strategy)

    @staticmethod
    def _prepare_row(row):
        text = row["text"]
        tokens = [text[token["span"]["start"]:token["span"]["end"]] for token in row["tokens"]]
        return tokens

    def _perform_strategy(self, gold, prediction, run_name, strategy):
        prediction_rel, gold_rel = strategy(gold, prediction)
        self._store[self.STRATS][run_name][self.GOLD_LABEL].extend(gold_rel)
        self._store[self.STRATS][run_name][self.PRED_LABEL].extend(prediction_rel)

    def extract_labels(self, gold, pred, exclude=["NO_MATCH"]):
        labels = set(gold)
        labels.update(pred)
        for ex in exclude:
            try:
                labels.remove(ex)
            except:
                pass  # If no entry exists, then everything is fine
        return list(labels)

    def _f1_score(self, run_name, average='weighted'):
        labels = self.extract_labels(self.gold_relations(run_name), self.pred_relations(run_name))
        return f1_score(self.gold_relations(run_name), self.pred_relations(run_name), average=average,
                        labels=labels)

    def _precision_score(self, run_name, average='weighted'):
        labels = self.extract_labels(self.gold_relations(run_name), self.pred_relations(run_name))
        return precision_score(self.gold_relations(run_name), self.pred_relations(run_name), average=average,
                               labels=labels)

    def _recall_score(self, run_name, average='weighted'):
        labels = self.extract_labels(self.gold_relations(run_name), self.pred_relations(run_name))
        return recall_score(self.gold_relations(run_name), self.pred_relations(run_name), average=average,
                            labels=labels)

    def _confusion_matrix(self, run_name):
        return pd.crosstab(pd.Series(self.gold_relations(run_name), name="Actual"),
                           pd.Series(self.pred_relations(run_name), name="Pred"))

    def gold_relations(self, run_name):
        return self._store[self.STRATS][run_name][self.GOLD_LABEL]

    def pred_relations(self, run_name):
        return self._store[self.STRATS][run_name][self.PRED_LABEL]

    def save_report(self):
        for run_name in self._store[self.STRATS]:
            for average in ["micro", "macro", "weighted"]:
                precision = self._precision_score(run_name, average=average)
                recall = self._recall_score(run_name, average=average)
                f1 = self._f1_score(run_name, average=average)

                self._store[self.STRATS][run_name]["precision_" + average] = precision
                self._store[self.STRATS][run_name]["recall_" + average] = recall
                self._store[self.STRATS][run_name]["f1_" + average] = f1

            conf_mat = self._confusion_matrix(run_name)
            self._save_confusion_matrix(conf_mat, run_name)
            self._store[self.STRATS][run_name]["confusion_matrix"] = "see " + run_name + ".png"

        file_name = ""
        if self._dir != "":
            file_name += self._dir + "/"
        file_name += "store" + self._file_suffix
        with open(file_name, "w") as outfile:
            json.dump(self._store, outfile)

    def load_report(self, file):
        with open(file, "r") as infile:
            data = json.load(infile)
            self._store = data

    def _save_confusion_matrix(self, df_confusion, to_file):
        file_name = ""
        if self._dir != "":
            file_name += self._dir + "/"
        file_name += to_file + ".png"
        sn.heatmap(df_confusion, annot=True)
        plt.savefig(file_name, bbox_inches='tight')
        plt.clf()


@click.command()
@click.option("--model-path", help="Path to model.tar.gz", required=True)
@click.option("--predictor", help="Fully qualified name of the predictor class", required=False, )
@click.option("--test-data", help="Path to test data in avro format", required=True)
@click.option("--include-package", "--inc", help="Like AllenNLP's inlcude-package", required=True)
@click.option("--output-dir", help="Output directory", required=True)
@click.option("--overwrite-dir", "-f", is_flag=True, default=False)
@click.option("--use_mock_predictor", "-ump", is_flag=True, default=False)
def smart_data_evaluate(model_path, predictor, test_data, include_package, output_dir, overwrite_dir,
                        use_mock_predictor):
    prepare_dir(output_dir, overwrite_dir)
    common_util.import_module_and_submodules(include_package)
    if "mock" in predictor.lower():
        use_mock_predictor = True
    if use_mock_predictor:
        splitted = predictor.split(".")
        mod = __import__(".".join(splitted[:-1]), fromlist=[splitted[-1]])
        klass = getattr(mod, splitted[-1])
        current_predictor_class = klass
    else:
        current_predictor_class = Predictor.by_name(predictor)

    @timeit
    def load_model(path):
        archive = load_archive(path)
        archive.model.eval()
        return current_predictor_class.from_archive(archive)

    @timeit
    def eval_model(predictor, test_data_path):
        evaluate_runner = EvaluationRunner(predictor, test_data_path, output_dir)
        evaluate_runner.evaluate(
                                 # MRE=respect_only_mandatory_args,
                                 # Cl=only_relation_classification,
                                 # CRE=all_args_mandatory,
                                 # AR=named_entity_recognition_v2,
                                 BRE=spert_only_two_mandatory_args,
                                 # MRE_no_trigger=respect_only_mandatory_args_no_trigger,
                                 # AR_no_trigger=named_entity_recognition_v2_no_trigger
        )
        evaluate_runner.save_report()

    if not use_mock_predictor:
        predictor = load_model(model_path)

    else:
        predictor = current_predictor_class(model_path)

    eval_model(predictor, test_data)


def prepare_dir(output_dir, overwrite_dir):
    folder_exists = os.path.isdir(output_dir)
    if folder_exists and not overwrite_dir:
        raise Exception(f"Folder \"{output_dir}\" already exists")
    if folder_exists and overwrite_dir:
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)


if __name__ == "__main__":
    smart_data_evaluate()
