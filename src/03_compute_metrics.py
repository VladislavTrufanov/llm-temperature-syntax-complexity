import csv

from pyconll.conllu import conllu

from common import PARSED_DIR, METRICS_DIR

# CSV header. build_row must produce fields in exactly this order.
METRICS_DATA = [
    'model', 
    'language', 
    'prompt_type', 
    'sent_id',
    'length', 
    'max_depth', 
    'mean_depth', 
    'mdd',
    'temperature', 
    'seed', 
    'record_id'
]


def get_depths(tokens, heads):
    """Return the depth of each token, counted as edges to the root."""
    depths = []
    for token in tokens:
        depth = 0
        current = int(token.id)
        while current != 0:
            current = heads[current]
            depth += 1
        depths.append(depth)
    return depths


def dependency_tokens(sentence):
    """Return tokens that have a head (drops multi-word tokens)."""
    return [t for t in sentence.tokens if t.head is not None]


def head_map(tokens):
    """Return a mapping from token id to head id."""
    return {int(t.id): int(t.head) for t in tokens}


def mean_dependency_distance(tokens, heads):
    """Return the mean distance between tokens and their heads.

    Root tokens (head == 0) are excluded. Returns None if there are
    no non-root tokens.
    """
    total = 0
    count = 0
    for token in tokens:
        head = heads[int(token.id)]
        if head != 0:
            total += abs(int(token.id) - head)
            count += 1
    if count == 0:
        return None
    return total / count


def compute_sentence_metrics(sentence):
    """Compute length, depth stats, and MDD for one sentence.

    Returns a dict with sent_id, length, max_depth, mean_depth, mdd,
    or None if the sentence has no head-bearing tokens or has no
    non-root tokens.
    """
    tokens = dependency_tokens(sentence)
    if not tokens:
        return None
    
    heads = head_map(tokens)
    depths = get_depths(tokens, heads)

    mdd = mean_dependency_distance(tokens, heads)
    if mdd is None:
        return None
    
    return {
        'sent_id': int(sentence.meta['sent_id']),
        'length': len(tokens),
        'max_depth': max(depths),
        'mean_depth': sum(depths) / len(depths),
        'mdd': mdd,
    }


def build_row(sentence, metrics):
    """Return a CSV row; field order must match METRICS_DATA."""
    return [
        sentence.meta['model'],
        sentence.meta['language'],
        sentence.meta['prompt_type'],
        metrics['sent_id'],
        metrics['length'],
        metrics['max_depth'],
        metrics['mean_depth'],
        metrics['mdd'],
        sentence.meta['temperature'],
        sentence.meta['seed'],
        sentence.meta['record_id'],
    ]


def process_file(path):
    """Load a CoNLL-U file and return metric rows, skipping sentences with no metrics."""
    rows = []
    doc = conllu.load_from_file(path)
    for sentence in doc:
        metrics = compute_sentence_metrics(sentence)
        if metrics is None:
            continue
        rows.append(build_row(sentence, metrics))
    return rows


def write_csv(rows, path):
    """Overwrite path with a CSV: METRICS_DATA header plus rows."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(METRICS_DATA)
        writer.writerows(rows)


def main():
    """Collect metrics from every .conllu in PARSED_DIR into metrics.csv."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in PARSED_DIR.glob('*.conllu'):
        rows.extend(process_file(file))
    write_csv(rows, METRICS_DIR / 'metrics.csv')
    print(len(rows))


if __name__ == '__main__':
    main()