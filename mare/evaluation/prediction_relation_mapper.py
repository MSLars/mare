
# Target Json Structure
# [{
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
#
#    ]
# }...]

# Predicted Relation structure
#
# [ Document
#   [ Sentence
#       [ Mult Events in a Sentence
#           [[<start_trigger_idx>, <end_trigger_idx>, <relation_name>], [<start_arg_n>, <end_arg_n>, <role_arg_n>]]
#       ...]
#   ]
# ]

def map_dygie(predictions):

    res = []
    for prediction in predictions:
        relations = []
        for sent in prediction["predicted_events"]:
            relation = None
            for event in sent:
                assert len(event) > 0
                relation = {
                    "name": event[0][2],
                    "ents": [
                        {"name": "trigger",
                         "start": event[0][0],
                         "end": event[0][1]}
                    ]
                }
                for argument in event[1:]:
                    relation["ents"] += [{
                        "name": argument[2],
                        "start": argument[0],
                        "end": argument[1]
                    }]
            if relation:
                relations += [relation]
        res += [{"relations": relations}]

    return res


def filter_current_gold(current_gold):
    for relation in current_gold["relations"]:
        trigger_ents = [r for r in relation['ents'] if r["name"] == "trigger"]
        if len(trigger_ents) != 1:
            return True
    return False