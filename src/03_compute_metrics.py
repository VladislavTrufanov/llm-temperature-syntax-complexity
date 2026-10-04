import csv
from pathlib import Path
from pyconll.conllu import conllu
from stanza.utils.conll import CoNLL

base_dir = Path(__file__).resolve().parent

project_root = base_dir.parent
parsed_proc = project_root / 'results' / 'parsed'
metrics_proc = project_root / 'results' / 'metrics'

metrics_proc.mkdir(parents=True, exist_ok=True) 

rows = []

for file in parsed_proc.glob('*.conllu'):
    metrics = conllu.load_from_file(file)
    
    for sentence in metrics:
        sent_id = int(sentence.meta['sent_id'])
        tokens = [t for t in sentence.tokens if t.head is not None]
            
        if not tokens:
            continue
            
        metrics_dict = {}
            
        for token in tokens:
            metrics_dict[int(token.id)] = int(token.head)

        depths = []

        for token in tokens:
            current = int(token.id)
            depth = 0
            while current != 0:
                current = metrics_dict[current]
                depth = depth + 1
            depths.append(depth)
        max_depth = max(depths)
        file_name = file.stem
        length = len(tokens)
        dd_sum = 0
        number = 0
        for token in tokens:
            head = metrics_dict[int(token.id)]
            if head != 0:
                dd_sum = dd_sum + abs(int(token.id) - int(head))
                number = number + 1
        if number == 0:
            continue
        mdd = dd_sum / number
        mean_depth = sum(depths) / len(depths)
        model = sentence.meta['model']
        language = sentence.meta['language']
        prompt_type = sentence.meta['prompt_type']
        temperature = sentence.meta['temperature']
        seed = sentence.meta['seed']
        record_id = sentence.meta['record_id']
        
        rows.append([model, language, prompt_type, sent_id, length, max_depth, mean_depth, mdd, temperature, seed, record_id])
        
with open(metrics_proc / 'metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model', 'language', 'prompt_type', 'sent_id', 'length', 'max_depth', 'mean_depth', 'mdd', 'temperature', 'seed', 'record_id'])
    for row in rows:
        writer.writerow(row)

print(len(rows))