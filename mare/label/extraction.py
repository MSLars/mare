import copy
import itertools
from typing import Iterable, List, Set, Tuple

from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedStringSpan, InvalidTagSequence

from mare.defs import relation_mandatory_args, replacable_args, relation_mandatory_args_without_trigger
from mare.utils import get_span_indizes


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

        n_m = max(n, self.end+1)

        tags = ["O"] * n_m

        for span in self.spans:

            if tags[span.start: span.end+1] != ["O"] * (1 + span.end - span.start):
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


def extract_relations_from_smart_sample(sample, only_mandatory=False, include_trigger=True):

    relations = []

    for relation in sample["relations"]:

        head = Span(relation["head_entity"]["token_indices"][0],
                    relation["head_entity"]["token_indices"][-1],
                    "head")

        tail = Span(relation["tail_entity"]["token_indices"][0],
                    relation["tail_entity"]["token_indices"][-1],
                    "tail")

        relations += [Relation([head, tail], label=relation["type"])]

    relations = sorted(relations, key=lambda r: (r.start, r.end, r.label))

    return relations


def combined_relation_tags(relations: Iterable[Relation], n: int, include_mode=False) -> List[str]:

    relations = sorted(relations, key=lambda r: (r.start, r.end, r.label))

    rel_types_mode = {l: True for l in {r.label for r in relations}}

    tags = ["O"] * n
    dominant_type = "O"
    if relations:
        dominant_type = relations[0].label

    relations = sorted(relations, key=lambda r: 0 if r.label == dominant_type else 1)

    def get_tag_type(t: str) -> str:
        if t == "O":
            return "O"
        rel_text = t[2:]
        type = rel_text[:rel_text.find("-")]
        return type

    for relation in relations:
        pre = rel_types_mode[relation.label]
        rel_types_mode[relation.label] = not rel_types_mode[relation.label]
        if include_mode:
            sub_tags = relation.get_bio_tags(n, mode="X") if pre else relation.get_bio_tags(n, mode="Y")
        else:
            sub_tags = relation.get_bio_tags(n)
        for i, tag in enumerate(sub_tags):
            typ = get_tag_type(tag)
            if tags[i] == "O" or (typ == dominant_type and get_tag_type(tags[i]) != dominant_type):
                if tag != "O":
                    tags[i] = tag

    return tags


def split_relation_spans(relation_spans: List[Span]) -> List[List[Span]]:
    """
    We try to find specific pattern in the relation roles that indicate multiple relation instead of one.

    1) E.g. person, organization, person, organization
        Here we have a "recurring structure" => two relations (person, organization), (person, organization)

    :param relation_spans:
    :return:
    """

    relation_roles = [s.role for s in relation_spans]
    relations_string = "".join([f"[{r}]" for r in relation_roles])

    prototypical_relation_roles = set()

    for role in relation_roles:
        if not role in prototypical_relation_roles:
            prototypical_relation_roles.add(role)
            continue

        # Here we found a possible reoccuring pattern
        pattern_size = len(prototypical_relation_roles)
        n = len(relation_roles)
        if n % pattern_size != 0:
            # pattern does not match
            break

        number_of_relations = n // pattern_size
        potential_split = []
        for i in range(number_of_relations):
            start = i * pattern_size
            end = start + pattern_size
            potential_split += [relation_spans[start: end]]

        for split in potential_split:
            split_roles = [s.role for s in split]
            if set(split_roles) != prototypical_relation_roles:
                break

        return potential_split

    return [relation_spans]


def xybio_tags_to_spans(
    tag_sequence: List[str], classes_to_ignore: List[str] = None
) -> List[TypedStringSpan]:
    classes_to_ignore = classes_to_ignore or []
    spans: Set[Tuple[str, Tuple[int, int]]] = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    active_pos_mode = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.

        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[4:]
        mode = string_tag[2] if len(string_tag) > 2 else "O"
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end), active_pos_mode))
            active_conll_tag = None
            active_pos_mode = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end), active_pos_mode))
            active_conll_tag = conll_tag
            active_pos_mode = mode
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end), active_pos_mode))
            active_conll_tag = conll_tag
            active_pos_mode = mode
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end),active_pos_mode))
    return list(spans)


def transform_spans_to_relation(spans: List[Span], max_inner_range=1000, has_mode=False, allow_overlaps=False, include_trigger=True):
    if include_trigger:
        mandatory_args = relation_mandatory_args
    else:
        mandatory_args = relation_mandatory_args_without_trigger

    relation_label = {a[0][:a[0].rfind("-")] for a in spans}

    relations = []

    for label in relation_label:

        modes = [""]
        if has_mode:
            modes = ["X", "Y"]

        for mode in modes:
            if mode != "":
                relation_spans = [Span(*a[1], role=a[0][a[0].rfind("-") + 1:]) for a in spans if
                                  a[0].startswith(label) and a[2] == mode]
            else:
                relation_spans = [Span(*a[1], role=a[0][a[0].rfind("-") + 1:]) for a in spans if
                                  a[0].startswith(label)]
            if relation_spans:
                final_spans = []
                start = 0
                for i in range(len(relation_spans) - 1):
                    curr_span_end = relation_spans[i].end
                    next_span_start = relation_spans[i + 1].start
                    curr_spans = relation_spans[start: i + 1]
                    curr_arg_types = {s.role for s in curr_spans}
                    if abs(curr_span_end - next_span_start) > max_inner_range and mandatory_args.get(label,
                                                                                                              set(
                                                                                                                      "UNLIKELY_LABEL_BAMMM")) <= curr_arg_types:
                        final_spans += [curr_spans]
                        start = i + 1

                final_spans += [relation_spans[start:]]

                for relation_spans in final_spans:

                    # Check if we have all mand. arguments
                    arg_types = {t.role for t in relation_spans}
                    # Check if we have found all mand. args
                    if label in mandatory_args and not mandatory_args[label] < arg_types:
                        missing_type = list(mandatory_args[label].difference(arg_types))

                        if len(missing_type) == 1:
                            missing_type = missing_type[0]
                            possible_types = replacable_args.get(f"{label}-{missing_type}", set())
                            help_relation = Relation(relation_spans, allow_overlaps=allow_overlaps)
                            potential_args = [s for s in spans if s[0] in possible_types]
                            potential_args = sorted(potential_args,
                                                    key=lambda s: abs(s[1][0] - help_relation.start) + abs(
                                                        s[1][1] - help_relation.end))

                            if potential_args:
                                new_span = Span(potential_args[0][1][0], potential_args[0][1][1], role=missing_type)
                                relation_spans += [new_span]

                    relations += [Relation(relation_spans, label=label, allow_overlaps=allow_overlaps)]

    return sorted(relations, key=lambda x: (x.start, x.end, x.label))


def transform_tags_to_relation(tags: List[str], max_inner_range=1000, has_mode=False, include_trigger=True):
    if has_mode:
        spans = sorted(xybio_tags_to_spans(tags), key=lambda x: x[1])
    else:
        spans = sorted(bio_tags_to_spans(tags), key=lambda x: x[1])

    return transform_spans_to_relation(spans, max_inner_range, has_mode, include_trigger=include_trigger)


def extract_entities(sample):

    entities = []
    for entity in sample["entities"]:

        entities += [Span(entity["token_indices"][0],
                          entity["token_indices"][-1],
                          "ent")]

    entities = sorted(entities, key=lambda x: (x.start, x.end))

    return entities


def combine_spans_to_entity_tags(spans: Span, n : int):

        ent_labels = ["O"] * n

        if spans is not None:
            for ent in spans:
                ent_labels[ent.start] = f"B-{ent.role}"
                if ent.end > ent.start:
                    ent_labels[ent.start + 1:ent.end + 1] = [f"I-{ent.role}"] * (ent.end - ent.start)

        return ent_labels


def transform_tag_to_entity(tags: List[str]):

    spans = [Span(s[1][0], s[1][1], role=s[0]) for s in bio_tags_to_spans(tags)]

    return sorted(spans, key=lambda  x : x.start)