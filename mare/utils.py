import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans


def compounding(start, stop, compound):
    def clip(value):
        return max(value, stop) if (start > stop) else min(value, stop)

    curr = float(start)
    while True:
        yield clip(curr)
        curr /= compound


def get_one_h_logits(logits, tags):
    one_h_logits = 0.0 * logits

    for batch in range(logits.shape[0]):
        for token_idx in range(logits.shape[1]):
            label_idx = tags[batch][token_idx] if len(tags[batch]) > token_idx else 1
            one_h_logits[batch][token_idx][label_idx] = 1.0

    return one_h_logits


def get_span_indizes(tags):
    return [sorted([s[1] for s in bio_tags_to_spans(ts)]) for ts in tags]


def get_span_tensor(spans, device="cpu"):

    lens = [len(s) for s in spans]
    num_max_spans = max(lens)

    spans = torch.tensor([pad_sequence_to_length(ts, num_max_spans, lambda: (0, 0)) for ts in spans])

    mask = torch.arange(num_max_spans).expand(len(lens), num_max_spans) < torch.tensor(lens).unsqueeze(1)

    return spans.to(device), mask.to(device)


def apply_attention(sequence, attention, mask):
    # Apply span wise attention
    batch_size = sequence.size(0)
    attended = sequence * 0.0

    if sequence.size(2) > 0:
        for i in range(sequence.size(1)):
            # (batch_size, max_num_spans)
            attention_scores = attention(sequence[:, i, :], sequence, mask)

            attended[:, i, :] = torch.bmm(attention_scores.view(batch_size, 1, -1), sequence).squeeze()

    return attended


def combine_sequence_with_spans(encoded, attended_spans, spans):
    combined = 0.0 * encoded
    for batch, spans in enumerate(spans):
        for span_index, span in enumerate(spans):
            s = span[0]
            e = span[1] + 1
            combined[batch, s:e, :] = attended_spans[batch, span_index, :].repeat(e - s).view(e - s, -1)

    return torch.cat((encoded, combined), 2)


def remove_line_split(word: str, transformer_res):

    if '­' in word[1:-1]:
        removed = f"{word[0]}{word[1:-1].replace('­', '')}{word[-1]}"
        if removed == transformer_res:
            return removed.strip()
    return word.strip()



if __name__ == "__main__":

    com = compounding(0.3, 0.01, 1.005)

    for i in range(1000):
        if i % 100 == 0:
            print(next(com))
        else:
            next(com)
