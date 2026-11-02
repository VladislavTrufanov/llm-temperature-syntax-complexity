import stanza
import json
import re
from itertools import product
from stanza.utils.conll import CoNLL

from common import dataset_stem, load_grid, LLM_OUTPUT_DIR, PARSED_DIR, LANG_CODES, METADATA_FIELDS

def extract_text(content):
    """Return plain text from an LLM output field.

    The field is either a string or a list of content blocks. From a
    list, only blocks with type "text" are joined (no separator).
    Returns "" for anything else, or if no text blocks are found.
    """
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
    """Remove bold, italic, and heading markers from text."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text


def parse_file(jsonl_path, out_path, nlp):
    """Parse each record in a JSONL file with Stanza and write CoNLL-U to out_path.

    Skips records whose text is empty after extraction and markdown
    stripping. Each sentence gets the record's METADATA_FIELDS as 
    CoNLL-U comments.
    """
    out_path.write_text('', encoding='utf-8')
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            text = strip_markdown(extract_text(record["output"]))
            if not text.strip():
                continue
            doc = nlp(text)
            current_metadata = {field: record[field] for field in METADATA_FIELDS}
            for sentence in doc.sentences:
                for key, value in current_metadata.items():
                    comment_string = f'{key} = {value}'
                    sentence.add_comment(comment_string)
            CoNLL.write_doc2conll(doc, str(out_path), 'a')


def main():
    """Parse LLM outputs with Stanza into CoNLL-U files."""
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    grid = load_grid()
    
    pipelines = {
        key: stanza.Pipeline(
            value,
            processors='tokenize,pos,lemma,depparse',
            download_method=None,
        )
        for key, value in LANG_CODES.items()
    }
    
    for i, (model, language, prompt_type) in enumerate(
        product(
            grid["models"],
            grid["languages"],
            grid["prompt_types"]
        ),
        start=1
    ):
        print(f"{i}: {model} {language} {prompt_type}", flush=True)
        stem = dataset_stem(model, language, prompt_type)
        parse_file(LLM_OUTPUT_DIR / f"{stem}.jsonl", PARSED_DIR / f"{stem}.conllu", pipelines[language])


if __name__ == "__main__":
    main()