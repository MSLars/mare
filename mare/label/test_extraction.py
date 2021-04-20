import pytest
import srsly
from allennlp.common.file_utils import cached_path

from mare.label.extraction import extract_relations_from_smart_sample, Relation, Span, combined_relation_tags, \
    transform_tags_to_relation, split_relation_spans, extract_entities, combine_spans_to_entity_tags, \
    transform_tag_to_entity
from mare.defs import relation_mandatory_args


@pytest.fixture
def dataset():
    smart_training_data_path = "https://fh-aachen.sciebo.de/s/MjcrDC3gDjwU7Vd/download"
    smart_validation_data_path = "https://fh-aachen.sciebo.de/s/3GpXCZLhjwm2SJU/download"
    smart_test_data_path = "https://fh-aachen.sciebo.de/s/9ghU4Qi1azUMFPW/download"

    data_paths = [smart_training_data_path, smart_validation_data_path, smart_test_data_path]

    dataset = []

    for file_path in data_paths:

        file_path = cached_path(file_path)

        with open(file_path, "r") as file:

            dataset += [srsly.json_loads(line) for line in file.readlines()]

    return dataset


def sort_gold(x):

    start = min(
        [a["conceptMention"]["span"]["start"] for a in x["args"] if a["role"] in relation_mandatory_args[x["name"]]])
    end = max(
        [a["conceptMention"]["span"]["end"] for a in x["args"] if a["role"] in relation_mandatory_args[x["name"]]])

    return start, end, x["name"]


def test_overlapping_entities():

    # Testcase 1: Reduce nested (complete included)

    relation = Relation(
        [Span(2, 2, "a"), Span(1, 3, "b")],
        label="Hallo"
    )

    reduced_relation = Relation(
        [Span(1, 3, "b")],
        label="Hallo"
    )

    assert relation == reduced_relation, f"Overlapping spans in relation not reduced!"

    # Testcase 2: Reduce nested (overlapping borders)

    relation = Relation(
        [Span(2, 5, "a"), Span(4, 8, "b")],
        label="Hallo"
    )

    reduced_relation = Relation(
        [Span(2, 5, "a")],
        label="Hallo"
    )

    assert relation == reduced_relation, f"Overlapping spans in relation not reduced!"

    # Testcase 3: one span with two roles

    relation = Relation(
        [Span(2, 2, "a"), Span(2, 2, "b")],
        label="Hallo"
    )

    reduced_relation = Relation(
        [Span(2, 2, "a")],
        label="Hallo"
    )

    assert relation == reduced_relation, f"Overlapping spans in relation not reduced!"

    #Testcase 4: Some complex sample

    relation = Relation(
        [Span(2, 2, "a"), Span(2, 2, "b"), Span(5, 7, "c"), Span(6, 8, "d"), Span(9, 11, "e")],
        label="Hallo"
    )

    reduced_relation = Relation(
        [Span(2, 2, "a"), Span(5, 7, "c"), Span(9, 11, "e")],
        label="Hallo"
    )

    assert relation == reduced_relation, f"Overlapping spans in relation not reduced!"


def test_extract_relations_from_smart_sample(dataset):

    for sample in dataset:

        extracted_relations = extract_relations_from_smart_sample(sample, only_mandatory=True)

        gold_relations = sample["relationMentions"]

        if len(extracted_relations) != len(gold_relations):
            assert False, f"Wrong number of Relations found in sample with id: {sample['id']}"

        gold_relations = sorted(gold_relations, key=sort_gold)

        for er, gr in zip(extracted_relations, gold_relations):

            assert gr["name"] == er.label


def test_relation_entities():

    spans = [Span(1, 1, role="a"),
        Span(3, 6, role="a"),
        Span(9, 9, role="a")]

    relation = Relation(list(spans),
        label="B"
    )

    entities = [Span(1, 1, role="B-a"),
        Span(3, 6, role="B-a"),
        Span(9, 9, role="B-a")]


    assert relation.spans == spans
    assert relation.entities == entities


def test_relation_tags():

    # Testcase 1: single argument

    relation = Relation(
        [Span(1, 1, role="a")],
        label="B"
    )

    tags = relation.get_bio_tags(4)
    gold = ["O", "B-B-a", "O", "O"]

    assert tags == gold, f"relation {relation} converted to sequence {tags} instead of {gold}"

    # Testcase 2: multiple argument

    relation = Relation(
        [Span(1, 1, role="a"),
        Span(3, 6, role="a"),
        Span(9, 9, role="a")],
        label="B"
    )

    tags = relation.get_bio_tags(12)
    gold = ["O", "B-B-a", "O", "B-B-a", "I-B-a", "I-B-a", "I-B-a", "O", "O", "B-B-a", "O", "O", ]

    assert tags == gold, f"relation {relation} converted to sequence {tags} instead of {gold}"

    # Testcase 3: invalid arguments

    with pytest.raises(RuntimeError):
        relation = Relation(
            [Span(1, 1, role="a"),
            Span(3, 6, role="a"),
            Span(4, 4, role="a"),
            Span(9, 9, role="a")],
            label="B"
        )

        tags = relation.get_bio_tags(12)

    # Testcase 4: wrong length

    relation = Relation(
        [Span(1, 1, role="a"),
        Span(3, 6, role="a"),
        Span(9, 9, role="a")],
        label="B"
    )
    n=8
    tags = relation.get_bio_tags(n)
    gold = ["O", "B-B-a", "O", "B-B-a", "I-B-a", "I-B-a", "I-B-a", "O",]
    assert tags == gold, f"relation {relation} converted to sequence {tags} instead of {gold} for n = {n}"

    # Testcase 5: Test mode inclusion

    relation = Relation(
        [Span(1, 1, role="a")],
        label="B"
    )

    tags = relation.get_bio_tags(4, mode="X")
    gold = ["O", "B-X-B-a", "O", "O"]

    assert tags == gold, f"relation {relation} converted to sequence {tags} instead of {gold}"


def test_combined_relation_tags():

    # Testcase 1: non overlapping

    relation_1 = Relation(
        [Span(1, 2, role="a"),
        Span(4, 4, role="b")],
        label="A"
    )
    relation_2 = Relation(
        [Span(6, 6, role="a"),
        Span(8, 9, role="b")],
        label="B"
    )
    n = 11
    tags = combined_relation_tags([relation_1, relation_2], n, include_mode=True)
    gold = ["O", "B-X-A-a", "I-X-A-a", "O", "B-X-A-b", "O", "B-X-B-a", "O", "B-X-B-b", "I-X-B-b", "O",]

    assert tags == gold, f"Combination of relations: {relation_1, relation_2} converted to sequence {tags} " \
                         f"instead of {gold} for n = {n}"

    # Testcase 2: overlapping and conflicting Relations, korrekt ordering
    relation_1 = Relation(
        [Span(1, 2, role="a"),
        Span(4, 4, role="b")],
        label="A"
    )
    relation_2 = Relation(
        [Span(3, 5, role="a"),
        Span(6, 6, role="a"),
        Span(8, 9, role="b")],
        label="B"
    )
    n = 11
    tags = combined_relation_tags([relation_1, relation_2], n, include_mode=True)
    gold = ["O", "B-X-A-a", "I-X-A-a", "B-X-B-a", "B-X-A-b", "I-X-B-a", "B-X-B-a", "O", "B-X-B-b", "I-X-B-b", "O",]

    assert tags == gold, f"Combination of relations: {relation_1, relation_2} converted to sequence {tags} " \
                         f"instead of {gold} for n = {n}"

    # Testcase 3: overlapping and conflicting Relations, wrong ordering

    relation_2 = Relation(
        [Span(3, 5, role="a"),
        Span(6, 6, role="a"),
        Span(8, 9, role="b")],
        label="B"
    )
    relation_1 = Relation(
        [Span(1, 2, role="a"),
        Span(4, 4, role="b")],
        label="A"
    )
    n = 11
    tags = combined_relation_tags([relation_2, relation_1], n)
    gold = ["O", "B-A-a", "I-A-a", "B-B-a", "B-A-b", "I-B-a", "B-B-a", "O", "B-B-b", "I-B-b", "O",]

    assert tags == gold, f"Combination of relations: {relation_1, relation_2} converted to sequence {tags} " \
                         f"instead of {gold} for n = {n}"

    # Testcase 4: Define fist relation type as dominant
    relation_1 = Relation(
        [Span(1, 2, role="a"),
        Span(4, 4, role="b")],
        label="A"
    )

    relation_2 = Relation(
        [Span(6, 6, role="a"),],
        label="B"
    )

    relation_3 = Relation(
        [Span(6, 6, role="b"), ],
        label="A"
    )

    n = 7
    tags = combined_relation_tags([relation_1, relation_2, relation_3], n, include_mode=True)
    gold = ["O", "B-X-A-a", "I-X-A-a", "O", "B-X-A-b", "O", "B-Y-A-b",]

    assert tags == gold, f"Combination of relations: {relation_1, relation_2} converted to sequence {tags} " \
                         f"instead of {gold} for n = {n}"

    # Testcase 5: Define fist relation type as dominant
    relation_1 = Relation(
        [Span(1, 2, role="a"),
        Span(4, 4, role="b")],
        label="B"
    )

    relation_2 = Relation(
        [Span(6, 6, role="a"),],
        label="A"
    )

    relation_3 = Relation(
        [Span(6, 6, role="b"), ],
        label="B"
    )

    n = 7
    tags = combined_relation_tags([relation_1, relation_2, relation_3], n, include_mode=True)
    gold = ["O", "B-X-B-a", "I-X-B-a", "O", "B-X-B-b", "O", "B-Y-B-b",]

    assert tags == gold, f"Combination of relations: {relation_1, relation_2} converted to sequence {tags} " \
                         f"instead of {gold} for n = {n}"

    # Testcase 5: Test_multiple alternating mode-relations

    relation_1 = Relation(
        [Span(1, 2, role="a"),
         Span(4, 4, role="b")],
        label="B"
    )

    relation_2 = Relation(
        [Span(6, 6, role="a"), Span(7, 7, role="b")],
        label="B"
    )

    relation_3 = Relation(
        [Span(9, 10, role="b"), ],
        label="B"
    )

    relation_4 = Relation(
        [Span(5, 5, role="d"), Span(8, 8, role="e")],
        label="A"
    )

    relation_5 = Relation(
        [Span(13, 14, role="d"), Span(16, 16, role="e")],
        label="A"
    )

    n = 17
    tags = combined_relation_tags([relation_1, relation_2, relation_3, relation_4, relation_5], n, include_mode=True)
    gold = ["O", "B-X-B-a", "I-X-B-a", "O", "B-X-B-b", "B-X-A-d", "B-Y-B-a", "B-Y-B-b", "B-X-A-e", "B-X-B-b", "I-X-B-b", "O", "O", "B-Y-A-d", "I-Y-A-d", "O", "B-Y-A-e", ]

    assert tags == gold, f"Combination of relations: {relation_1, relation_2} converted to sequence {tags} " \
                         f"instead of {gold} for n = {n}"


def test_split_relation_spans():

    # Testcase 1: Simple Split

    relation_spans = [Span(1, 1, "a"), Span(3, 3, "b"), Span(4, 4, "a"), Span(5, 1 , "b"),]

    splitted_relation_spans = split_relation_spans(relation_spans)
    gold_splitted = [
        [Span(1, 1, "a"), Span(3, 3, "b")],
        [Span(4, 4, "a"), Span(5, 1, "b")]
    ]

    assert splitted_relation_spans == gold_splitted

    # Testcase 2: No Split

    relation_spans = [Span(1, 1, "a"), Span(3, 3, "b"), Span(4, 4, "c"),]

    splitted_relation_spans = split_relation_spans(relation_spans)
    gold_splitted = [
        [Span(1, 1, "a"), Span(3, 3, "b"), Span(4, 4, "c")],
    ]

    assert splitted_relation_spans == gold_splitted

    # Testcase 3: No Split but some repitition

    relation_spans = [Span(1, 1, "a"), Span(3, 3, "b"), Span(4, 4, "c"), Span(6, 6, "a"), Span(7, 7, "b"),]

    splitted_relation_spans = split_relation_spans(relation_spans)
    gold_splitted = [
        [Span(1, 1, "a"), Span(3, 3, "b"), Span(4, 4, "c"), Span(6, 6, "a"), Span(7, 7, "b"),],
    ]

    assert splitted_relation_spans == gold_splitted

    # Testcase 4: Split complicated

    relation_spans = [Span(1, 1, "a"), Span(3, 3, "b"), Span(4, 4, "c"),
                      Span(6, 6, "a"), Span(7, 7, "c"), Span(9, 9, "b"),
                      Span(13, 13, "c"), Span(15, 15, "b"), Span(17, 17, "a"),]

    splitted_relation_spans = split_relation_spans(relation_spans)
    gold_splitted = [
        [Span(1, 1, "a"), Span(3, 3, "b"), Span(4, 4, "c")],
        [Span(6, 6, "a"), Span(7, 7, "c"), Span(9, 9, "b")],
        [Span(13, 13, "c"), Span(15, 15, "b"), Span(17, 17, "a"),],
    ]

    assert splitted_relation_spans == gold_splitted


def test_transform_label_to_relation():

    # Testcase 1: Multiple Relation instances

    tags = ["O", "B-X-Ol-per", "O", "B-X-Ol-org", "O", "B-Y-Ol-per", "I-Y-Ol-per", "O", "B-Y-Ol-org", "O", ]

    relations = transform_tags_to_relation(tags, has_mode=True)

    gold = [
        Relation([Span(1, 1, "per"), Span(3, 3, "org")], label="Ol"),
        Relation([Span(5, 6, "per"), Span(8, 8, "org")], label="Ol"),
    ]

    assert relations == gold, f"Tags Sequence {tags}"

    # Testcase 2: Shared Argument

    tags = ["O", "B-X-Accident-location", "O", "B-X-Accident-trigger", "O", "B-X-TrafficJam-trigger", "O"]

    relations = transform_tags_to_relation(tags, has_mode=True)

    gold = [
        Relation([Span(1, 1, "location"), Span(3, 3, "trigger")], label="Accident"),
        Relation([Span(1, 1, "location"), Span(5, 5, "trigger")], label="TrafficJam"),
    ]

    assert relations == gold, f"Tags Sequence {tags}"

    # Testcase 3: We expect Relations to have an inner-distance from at most 18

    tags = ["O", "B-X-Accident-location", "O", "B-X-Accident-trigger", "O",
            "O", "B-Y-Accident-location", "O", "B-Y-Accident-trigger", "O",
            "O", "O", "O", "O", "O", "O", "O", "O", "O", "O",
            "O", "O", "O", "O", "O", "O", "O", "O", "O", "O",
            "O", "B-X-Accident-location", "O", "B-X-Accident-trigger", "O", ]

    relations = transform_tags_to_relation(tags, has_mode=True, max_inner_range=15)

    gold = [
        Relation([Span(1, 1, "location"), Span(3, 3, "trigger")], label="Accident"),
        Relation([Span(6, 6, "location"), Span(8, 8, "trigger")], label="Accident"),
        Relation([Span(31, 31, "location"), Span(33, 33, "trigger")], label="Accident")
    ]

    assert relations == gold, f"Tags Sequence {tags}"


def test_re_transform_relations_keep_first(dataset):

    number_of_samples_with_relations = 0
    number_of_correct_samples = 0

    tp = 0
    fp = 0
    fn = 0

    for sample in dataset:

        n = len(sample["tokens"])

        gold_relations = extract_relations_from_smart_sample(sample, only_mandatory=False)
        gold_relations = sorted(gold_relations, key=lambda x: (x.start, x.end, x.label))
        tags = combined_relation_tags(gold_relations, n, include_mode=False)

        transformed_relations = transform_tags_to_relation(tags, max_inner_range=10, has_mode=False)

        if not gold_relations:
            assert not bool(transformed_relations), f"Should not have found any relation for" \
                                                f"\n{gold_relations}" \
                                                f"\nbut found:" \
                                                f"\n{transformed_relations}"
            number_of_correct_samples += 1
        else:

            # assert gold_relations[0].label == transformed_relations[0].label, f"Transformation of \n{gold_relations}" \
            #                                                                   f"\n{tags}" \
            #                                                                   f"\n{transformed_relations}" \
            #                                                                   f"\n first relation has not same label."

            number_of_samples_with_relations += 1

            gold_set = set(gold_relations)
            transformed_set = set(transformed_relations)

            if gold_set == transformed_set:
                number_of_correct_samples += 1

            # if gold_set != transformed_set:
            #     print("hallo")

            tp += len(gold_set.intersection(transformed_set))
            fn += len(gold_set.difference(transformed_set))
            fp += len(transformed_set.difference(gold_set))


    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    print(f"f: {f} p: {p} r: {r} of first relations correct!")
    print(f"Amount of correct samples {number_of_correct_samples / len(dataset)}")

    assert f > 0.85, "F Score to low"


def test_ner_extraction(dataset):
    for sample in dataset:

        concept_mentions = sample["conceptMentions"]
        n = len(sample["tokens"])
        gold_entities = extract_entities(sample)

        assert len(concept_mentions) == len(gold_entities), f"Found unequal number of entities in {sample}"

        ent_tags = combine_spans_to_entity_tags(gold_entities, n)

        transformed_entities = transform_tag_to_entity(ent_tags)
