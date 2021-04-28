import copy
import itertools
from typing import List, Iterable

relation_mandatory_args = {
    "Accident": {"location", "trigger"},
    "CanceledRoute": {"location", "trigger"},
    "CanceledStop": {"location", "trigger"},
    "Delay": {"location", "trigger"},
    "Disaster": {"type", "location"},
    "Obstruction": {"location", "trigger"},
    "RailReplacementService": {"location", "trigger"},
    "TrafficJam": {"location", "trigger"},
    "Acquisition": {"buyer", "acquired"},
    "Insolvency": {"company", "trigger"},
    "Layoffs": {"company", "trigger"},
    "Merger": {"old", "new"},
    "OrganizationLeadership": {"organization", "person"},
    "SpinOff": {"parent", "child"},
    "Strike": {"company", "trigger"}
}

class Span:
    def __init__(self, pos1, pos2, role: str = None, text: str = None):
        self.span = (int(pos1), int(pos2))
        self.role = role
        self.text = text

    @property
    def start(self):
        return self.span[0]

    @property
    def end(self):
        return self.span[1]

    def to_josn(self):
        return {
            "start": self.span[0],
            "end": self.span[1],
            "name": self.role if self.role else "",
        }

    def __repr__(self):
        return f"[{repr(self.span)}, {self.role if self.role is not None else ''}]"

    def __eq__(self, other):
        return self.role == other.role and self.span == other.span

    def __hash__(self):
        return hash((self.span, self.role))


class Relation:
    def __init__(self, spans: List[Span], label: str = None, allow_overlaps=False):

        if not len(spans) >= 1:
            raise AttributeError("Tried to create Relation with no spans")

        # TODO ist das gut so? Review
        if not allow_overlaps:
            for s1, s2 in itertools.combinations(spans, 2):
                if s2.start <= s1.start <= s2.end or s2.start <= s1.end <= s2.end:
                    # Found conflicting spans!
                    if s1.start < s2.start:
                        spans.remove(s2)
                    elif s2.start < s1.start:
                        spans.remove(s1)
                    elif s1.end < s2.end:
                        spans.remove(s2)
                    elif s2.end < s1.end:
                        spans.remove(s1)
                    elif s1.role < s2.role:
                        spans.remove(s2)
                    else:
                        spans.remove(s1)

        self.spans = spans
        self.label = label

    @property
    def start(self):
        return min((s.start for s in self.spans))

    @property
    def end(self):
        return max((s.end for s in self.spans))

    def get_bio_tags(self, n: int, mode: str = None) -> Iterable[str]:

        if not mode:
            mode = ""

        n_m = max(n, self.end + 1)

        tags = ["O"] * n_m

        for span in self.spans:

            if tags[span.start: span.end + 1] != ["O"] * (1 + span.end - span.start):
                raise RuntimeError(f"Nested argument types for relation {self}, "
                                   f"cannot build well-defined tag sequence.")

            tag_post = f"{self.label}-{span.role}"
            if mode != "":
                tag_post = f"{mode}-{tag_post}"
            tags[span.start] = f"B-{tag_post}" if mode == "" else f"B-{tag_post}"
            if span.end > span.start:
                tags[span.start + 1:span.end + 1] = [f"I-{tag_post}"] * (span.end - span.start)

        return tags[:n]

    @property
    def entities(self):
        entities = copy.deepcopy(self.spans)

        for ent, span in zip(entities, self.spans):
            ent.role = f"{self.label}-{span.role}"

        return entities

    def to_json(self):
        return {
            "name": self.label,
            "ents": [s.to_josn() for s in self.spans],
        }

    def __repr__(self):
        return f"{{({repr(self.spans)}, {self.label if self.label is not None else ''})}}"

    def __eq__(self, other):
        return {s for s in self.spans} == {s for s in other.spans} and self.label == other.label

    def __hash__(self):
        return hash(self.label)


def extract_relations_from_smart_sample(sample, only_mandatory=False, include_trigger=True):
    token_borders = [(t["span"]["start"], t["span"]["end"]) for t in sample["tokens"]]

    relations = []

    for relation in sample["relationMentions"]:

        relation_name = relation["name"]
        spans = []

        for argument in relation["args"]:

            if only_mandatory and argument["role"] not in relation_mandatory_args[relation_name]:
                continue

            if not include_trigger and argument["role"] == "trigger":
                continue

            argument_role = argument["role"]
            entity = argument["conceptMention"]["span"]
            start, end = get_token_index_from_char_index(entity["start"], entity["end"], token_borders)

            spans += [Span(start, end, role=argument_role)]

        if spans:
            relations += [Relation(spans, label=relation_name)]

    relations = sorted(relations, key=lambda r: (r.start, r.end, r.label))

    return relations


def get_token_index_from_char_index(start, end, token_borders, fuzzy=False):
    token_borders = sorted(token_borders, key=lambda x: x[0])

    if fuzzy:
        token_start = [i[0] for i in enumerate(token_borders) if i[1][0] >= start]
        token_end = [i[0] for i in enumerate(token_borders) if i[1][1] >= end]
    else:
        token_start = [i[0] for i in enumerate(token_borders) if i[1][0] == start]
        token_end = [i[0] for i in enumerate(token_borders) if i[1][1] == end]

    try:
        return token_start[0], token_end[0]
    except IndexError:
        raise AttributeError(f"Tokenborders {token_borders} are not valid "
                             f"for char indizies {start, end} with fuzzy={fuzzy}")

def extract_entities(sample):
    token_borders = [(t["span"]["start"], t["span"]["end"]) for t in sample["tokens"]]

    entities = []
    for concept_mention in sample["conceptMentions"]:
        start = concept_mention["span"]["start"]
        end = concept_mention["span"]["end"]

        start = [i[0] for i in enumerate(token_borders) if i[1][0] == start]
        end = [i[0] for i in enumerate(token_borders) if i[1][1] == end]

        if len(start) > 0 and len(end) > 0:
            entities += [Span(start[0], end[0], role=concept_mention["type"])]

    return entities
