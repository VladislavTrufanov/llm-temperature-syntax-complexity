import csv

from pyconll.conllu import conllu

from common import PARSED_DIR, METRICS_DIR

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
    depths = []
    for token in tokens:
        depth = 0
        current = int(token.id)
        while current != 0:
            current = heads[current]
            depth += 1
        depths.append(depth)
    return depths

def compute_sentence_metrics(sentence):
    sent_id = int(sentence.meta['sent_id'])
    tokens = [t for t in sentence.tokens if t.head is not None]
    if not tokens:
        return None

    heads = {int(t.id): int(t.head) for t in tokens}
    depths = get_depths(tokens, heads)

    length = len(tokens)
    max_depth = max(depths)
    mean_depth = sum(depths) / len(depths)

    dd_sum = 0
    number = 0
    for token in tokens:
        head = heads[int(token.id)]
        if head != 0:
            dd_sum += abs(int(token.id) - head)
            number += 1
    if number == 0:
        return None

    mdd = dd_sum / number

    return {
        'sent_id': sent_id,
        'length': length,
        'max_depth': max_depth,
        'mean_depth': mean_depth,
        'mdd': mdd,
    }

def build_row(sentence, metrics):
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
    rows = []
    doc = conllu.load_from_file(path)
    for sentence in doc:
        metrics = compute_sentence_metrics(sentence)
        if metrics is None:
            continue
        rows.append(build_row(sentence, metrics))
    return rows

def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(METRICS_DATA)
        writer.writerows(rows)

def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in PARSED_DIR.glob('*.conllu'):
        rows.extend(process_file(file))
    write_csv(rows, METRICS_DIR / 'metrics.csv')
    print(len(rows))

if __name__ == '__main__':
    main()