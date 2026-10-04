import stanza
import json
import re
from pathlib import Path
from stanza.utils.conll import CoNLL

base_dir = Path(__file__).resolve().parent

project_root = base_dir.parent
output_proc = project_root / 'results' / 'llm_output' 
parsed_proc = project_root / 'results' / 'parsed'

parsed_proc.mkdir(parents=True, exist_ok=True)

nlp_fr = stanza.Pipeline('fr', processors='tokenize,pos,lemma,depparse', download_method=None)
nlp_en = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', download_method=None)

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""

def strip_markdown(text):
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text

def parse_file(jsonl_path, nlp, out_dir):
    out_path = out_dir / f"{jsonl_path.stem}.conllu"
    out_path.write_text('', encoding='utf-8')

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            text = extract_text(record["output"])
            if not text.strip():
                continue
            text = strip_markdown(text)
            doc = nlp(text)
            for sentence in doc.sentences:
                sentence.add_comment(f"model = {record['model']}")
                sentence.add_comment(f"language = {record['language']}")
                sentence.add_comment(f"prompt_type = {record['prompt_type']}")
                sentence.add_comment(f"temperature = {record['temperature']}")
                sentence.add_comment(f'seed = {record["seed"]}')
                sentence.add_comment(f'record_id = {record["record_id"]}')
            CoNLL.write_doc2conll(doc, str(out_path), 'a')

for file in output_proc.glob('*french*'):
    parse_file(file, nlp_fr, parsed_proc)

for file in output_proc.glob('*english*'):
    parse_file(file, nlp_en, parsed_proc)