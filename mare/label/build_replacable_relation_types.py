import itertools

import srsly
from allennlp.common.file_utils import cached_path

from mare.label.extraction import extract_relations_from_smart_sample

if __name__ == "__main__":

    smart_training_data_path = "https://fh-aachen.sciebo.de/s/MjcrDC3gDjwU7Vd/download"

    dataset = []

    file_path = cached_path(smart_training_data_path)

    with open(file_path, "r") as file:

        dataset += [srsly.json_loads(line) for line in file.readlines()]

    replacable = {}

    for sample in dataset:
        gold_relations = extract_relations_from_smart_sample(sample, only_mandatory=True)

        if len(gold_relations) < 2:
            continue

        n = len(sample["tokens"])

        for r1, r2 in itertools.combinations(gold_relations, 2):

            tags1 = r1.get_bio_tags(n)
            tags2 = r2.get_bio_tags(n)

            for t1, t2 in zip(tags1, tags2):

                if t1 != "O" and t2 != "O":
                    replacable[t1[2:]] = replacable.get(t1[2:], set([t1[2:]]))
                    replacable[t1[2:]].add(t2[2:])
                    replacable[t2[2:]] = replacable.get(t2[2:], set([t2[2:]]))
                    replacable[t2[2:]].add(t1[2:])

    print(replacable)