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
#                     "mandatory": true (optional, just for gold)
#                 }
#             ]
#         }
#     ]
# }
import copy

from mare.defs import relation_mandatory_args

NO_MATCH = "NO_MATCH"


def sort_entities_of_rel(relation):
    return {"name": relation["name"], "ents": sort_entities(relation["ents"])}


def sort_entities(ents):
    return sorted(ents, key=lambda x: x["start"])


def sort_relations(rels):
    return sorted(rels, key=lambda x: x["name"])


def reduce_to_mandatory_arguments(relation):
    return {"name": relation["name"],
            "ents": sort_entities(
                [ele for ele in relation["ents"] if
                 ele["name"] in relation_mandatory_args[relation["name"]]])}


def reduce_relations_to_mandatory_arguments(relations):
    rels_reduced = []
    for rel in relations:
        rel_reduced = reduce_to_mandatory_arguments(rel)
        rels_reduced.append(rel_reduced)
    return rels_reduced


def copy_dict(source):
    return copy.deepcopy(source)


def named_entity_recognition(gold, prediction):
    gold_copy = copy_dict(gold)
    pred_copy = copy_dict(prediction)

    gold_ner = []
    pred_ner = []

    gold_ents = [ent for rel in gold_copy["relations"] for ent in rel["ents"]]
    pred_ents = [ent for rel in pred_copy["relations"] for ent in rel["ents"]]

    for pred_ent in pred_ents:
        if pred_ent in gold_ents:
            pred_ner.append(pred_ent["name"])
            gold_ner.append(pred_ent["name"])
            gold_ents.remove(pred_ent)
        else:
            pred_ner.append(pred_ent["name"])
            gold_ner.append(NO_MATCH)

    for gold_ent in gold_ents:
        pred_ner.append(NO_MATCH)
        gold_ner.append(gold_ent["name"])

    return pred_ner, gold_ner


def named_entity_recognition_v2(gold, prediction):
    gold_copy = copy_dict(gold)
    pred_copy = copy_dict(prediction)

    gold_ner = []
    pred_ner = []

    def extract_ents(relations):
        ents = []
        for rel in relations:
            for ent in rel["ents"]:
                ent["name"] = f"{rel['name']}-{ent['name']}"
                ents.append(ent)
        return ents
    gold_ents = extract_ents(gold_copy["relations"])
    pred_ents = extract_ents(pred_copy["relations"])

    for pred_ent in pred_ents:
        if pred_ent in gold_ents:
            pred_ner.append(pred_ent["name"])
            gold_ner.append(pred_ent["name"])
            gold_ents.remove(pred_ent)
        else:
            pred_ner.append(pred_ent["name"])
            gold_ner.append(NO_MATCH)

    for gold_ent in gold_ents:
        pred_ner.append(NO_MATCH)
        gold_ner.append(gold_ent["name"])

    return pred_ner, gold_ner


def named_entity_recognition_v2_no_trigger(gold, prediction):
    gold_copy = copy_dict(gold)
    pred_copy = copy_dict(prediction)

    gold_ner = []
    pred_ner = []

    def extract_ents(relations):
        ents = []
        for rel in relations:
            for ent in rel["ents"]:
                if ent["name"].lower() != "trigger":
                    ent["name"] = f"{rel['name']}-{ent['name']}"
                    ents.append(ent)
        return ents

    gold_ents = extract_ents(gold_copy["relations"])
    pred_ents = extract_ents(pred_copy["relations"])

    for pred_ent in pred_ents:
        if pred_ent in gold_ents:
            pred_ner.append(pred_ent["name"])
            gold_ner.append(pred_ent["name"])
            gold_ents.remove(pred_ent)
        else:
            pred_ner.append(pred_ent["name"])
            gold_ner.append(NO_MATCH)

    for gold_ent in gold_ents:
        pred_ner.append(NO_MATCH)
        gold_ner.append(gold_ent["name"])

    return pred_ner, gold_ner


def only_relation_classification(gold, prediction):
    """
    Returns prediction_rel, gold_rel
    """
    gold_rel_names = [rel["name"] for rel in gold["relations"]]
    prediction_rel_names = [rel["name"] for rel in prediction["relations"]]

    prediction_relations = []
    gold_relations = []

    for pred in prediction_rel_names:
        if pred in gold_rel_names:
            prediction_relations.append(pred)
            gold_relations.append(pred)
            gold_rel_names.remove(pred)
        else:
            prediction_relations.append(pred)
            gold_relations.append(NO_MATCH)

    for gold_rel in gold_rel_names:
        prediction_relations.append(NO_MATCH)
        gold_relations.append(gold_rel)

    return prediction_relations, gold_relations


def respect_only_mandatory_args(gold, prediction):
    gold_copy = copy_dict(gold)
    prediction_copy = copy_dict(prediction)

    prediction_relations = []
    gold_relations = []

    gold_rels_reduced = []
    for gold_rel in gold_copy["relations"]:
        gold_rel_reduced = reduce_to_mandatory_arguments(gold_rel)
        gold_rels_reduced.append(gold_rel_reduced)

    for pred in prediction_copy["relations"]:
        pred_reduced = reduce_to_mandatory_arguments(pred)

        if pred_reduced in gold_rels_reduced:
            prediction_relations.append(pred_reduced["name"])
            gold_relations.append(pred_reduced["name"])
            gold_rels_reduced.remove(pred_reduced)
        else:
            prediction_relations.append(pred_reduced["name"])
            gold_relations.append(NO_MATCH)

    for gold_rel in gold_rels_reduced:
        prediction_relations.append(NO_MATCH)
        gold_relations.append(gold_rel["name"])

    return prediction_relations, gold_relations


def filter_role(rel, role="trigger"):
    new_rel = {"name": rel["name"]}
    new_ents = []
    for ent in rel["ents"]:
        if ent["name"].lower() != role:
            new_ent = {"name": ent["name"],
                       "start": ent["start"],
                        "end": ent["end"]
                       }
            new_ents.append(new_ent)
    new_rel["ents"] = new_ents
    return new_rel


def respect_only_mandatory_args_no_trigger(gold, prediction):
    gold_copy = copy_dict(gold)
    prediction_copy = copy_dict(prediction)

    prediction_relations = []
    gold_relations = []

    gold_rels_reduced = []
    for gold_rel in gold_copy["relations"]:
        gold_rel_reduced = filter_role(reduce_to_mandatory_arguments(gold_rel))
        gold_rels_reduced.append(gold_rel_reduced)

    for pred in prediction_copy["relations"]:
        pred_reduced = filter_role(reduce_to_mandatory_arguments(pred))

        if pred_reduced in gold_rels_reduced:
            prediction_relations.append(pred_reduced["name"])
            gold_relations.append(pred_reduced["name"])
            gold_rels_reduced.remove(pred_reduced)
        else:
            prediction_relations.append(pred_reduced["name"])
            gold_relations.append(NO_MATCH)

    for gold_rel in gold_rels_reduced:
        prediction_relations.append(NO_MATCH)
        gold_relations.append(gold_rel["name"])

    return prediction_relations, gold_relations

def spert_only_two_mandatory_args(gold, prediction):
    gold_copy = copy_dict(gold)
    prediction_copy = copy_dict(prediction)

    prediction_relations = []
    gold_relations = []

    gold_rels_tmp = []
    for gold_rel in gold_copy["relations"]:
        gold_rel_reduced = reduce_to_mandatory_arguments(gold_rel)
        gold_rels_tmp.append(gold_rel_reduced)

    if sum([len(rel["ents"]) != 2 for rel in gold_rels_tmp]) > 0:
        return [], []

    gold_rels_reduced = [rel for rel in gold_rels_tmp if len(rel["ents"]) == 2]


    for pred in prediction_copy["relations"]:
        pred_reduced = reduce_to_mandatory_arguments(pred)

        if pred_reduced in gold_rels_reduced:
            prediction_relations.append(pred_reduced["name"])
            gold_relations.append(pred_reduced["name"])
            gold_rels_reduced.remove(pred_reduced)
        else:
            prediction_relations.append(pred_reduced["name"])
            gold_relations.append(NO_MATCH)

    for gold_rel in gold_rels_reduced:
        prediction_relations.append(NO_MATCH)
        gold_relations.append(gold_rel["name"])

    return prediction_relations, gold_relations


def all_args_mandatory(gold, prediction):
    gold_copy = copy_dict(gold)
    prediction_copy = copy_dict(prediction)

    prediction_relations = []
    gold_relations = []

    gold_rels_sorted = []
    for gold_rel in gold_copy["relations"]:
        gold_rel_sorted = sort_entities_of_rel(gold_rel)
        gold_rels_sorted.append(gold_rel_sorted)

    for pred in prediction_copy["relations"]:
        pred_sorted = sort_entities_of_rel(pred)

        if pred_sorted in gold_rels_sorted:
            prediction_relations.append(pred_sorted["name"])
            gold_relations.append(pred_sorted["name"])
            gold_rels_sorted.remove(pred_sorted)
        else:
            prediction_relations.append(pred_sorted["name"])
            gold_relations.append(NO_MATCH)

    for gold_rel in gold_rels_sorted:
        prediction_relations.append(NO_MATCH)
        gold_relations.append(gold_rel["name"])

    return prediction_relations, gold_relations


if __name__ == "__main__":
    gold = {
        "relations": [
            {
                "name": "Obstruction",
                "ents": [
                    {
                        "name": "location",
                        "start": 0,
                        "end": 0
                    },
                    {
                        "name": "trigger",
                        "start": 5,
                        "end": 5
                    },
                    {
                        "name": "something_different",
                        "start": 10,
                        "end": 5
                    }
                ]
            }
        ]
    }
    prediction = {
        "relations": [
            {
                "name": "Obstruction",
                "ents": [
                    {
                        "name": "location",
                        "start": 0,
                        "end": 0
                    },
                    {
                        "name": "trigger",
                        "start": 5,
                        "end": 5
                    },
                    {
                        "name": "something_different",
                        "start": 10,
                        "end": 5
                    }
                ]
            }
        ]
    }

    respect_only_mandatory_args(gold, prediction)
