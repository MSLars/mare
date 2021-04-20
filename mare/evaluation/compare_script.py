import srsly

def compare(fileA, fileB, source_file):
    df_A, df_B, source, proc_src = prepare_datasets(fileA, fileB, source_file)
    equal, not_equal = extract_relevant_entries(df_A, df_B, fileA, fileB, proc_src)

    #entries_with_rels = [entry for entry in equal if len(entry["entry"]["relations"]["relations"]) > 0]
    relevant_ids = [entry["id"] for entry in not_equal]
    #relevant_ids += [entry["id"] for entry in random.sample(entries_with_rels, 10)]

    filtered_src = [entry for entry in source if entry["id"] in relevant_ids]

    srsly.write_jsonl("smart_data_test_debug_v4.jsonl", filtered_src)
    srsly.write_jsonl("equal_v4.txt", equal)
    srsly.write_jsonl("not_equal_v4.txt", not_equal)


def prepare_datasets(fileA, fileB, source_file):
    df_A = list(srsly.read_jsonl(fileA))
    df_B = list(srsly.read_jsonl(fileB))
    source = list(srsly.read_jsonl(source_file))
    proc_src = []
    for entry in source:
        proc_src.append({"tokens": row_to_tokens(entry), "id": entry["id"]})
    return df_A, df_B, source, proc_src


def row_to_tokens(source):
    text = source["text"]
    tokens = [text[t["span"]["start"]: t["span"]["end"]] for t in source["tokens"]]
    return tokens


def extract_relevant_entries(df_A, df_B, fileA, fileB, proc_src):
    equal = []
    not_equal = []
    for entry in df_A:
        matched_entry = [entry_b for entry_b in df_B if entry["tokens"] == entry_b["tokens"]]
        matched_source = [entry_source for entry_source in proc_src if entry["tokens"] == entry_source["tokens"]]
        assert len(matched_entry) == 1
        assert len(matched_source) == 1

        if matched_entry[0] != entry:
            not_equal.append({"id": matched_source[0]["id"],
                              fileA: entry,
                              fileB: matched_entry[0]})
        else:
            equal.append({"id": matched_source[0]["id"],
                          "entry": entry})
    return equal, not_equal


if __name__ == "__main__":
    a = "example_v4.jsonl"
    b = "example_4_lars.log"
    source = "../../data/smart_data/smart_data_test.jsonl"
    compare(a, b, source)