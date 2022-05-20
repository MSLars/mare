import random

import srsly

if __name__ == "__main__":

    corpus = list(srsly.read_jsonl("data/sample.jsonl"))

    random.shuffle(corpus)

    train = corpus[:3200]
    dev = corpus[3200: 4100]
    test = corpus[4100:]

    srsly.write_jsonl("data/re_train.jsonl", train)
    srsly.write_jsonl("data/re_dev.jsonl", dev)
    srsly.write_jsonl("data/re_test.jsonl", test)