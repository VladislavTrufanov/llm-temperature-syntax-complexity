import json

import pandas as pd

from common import LLM_OUTPUT_DIR, METRICS_DIR

# According to the present project design
EXPECTED_TEXTS = 144


def read_sentence_metrics():
    return pd.read_csv(METRICS_DIR / "metrics.csv")


def make_text_level(sentences):
    """Aggregate sentence metrics to one row per text (record_id)."""
    grouped = sentences.groupby("record_id")
    text_level = grouped.agg(
        # These columns are the same in all sentences of a text, so we take the first value.
        model=("model", "first"),
        language=("language", "first"),
        prompt_type=("prompt_type", "first"),
        temperature=("temperature", "first"),
        seed=("seed", "first"),
        n_sent=("sent_id", "count"),
        length_med=("length", "median"),
        max_depth_med=("max_depth", "median"),
        mean_depth_med=("mean_depth", "median"),
        mdd_med=("mdd", "median"),
    )
    return text_level.reset_index()


def read_llm_fields():
    """Read every .jsonl output, keeping only record_id, finish_reason, completion_tokens."""
    rows = []
    for path in sorted(LLM_OUTPUT_DIR.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                rows.append({
                    "record_id": record["record_id"],
                    "finish_reason": record["finish_reason"],
                    "completion_tokens": record["completion_tokens"],
                })
    return pd.DataFrame(rows)


def join_tables(text_level, llm_fields):
    """Attach finish_reason and completion_tokens to each text by record_id.

    Left join keeps all texts from text_level. Texts that lost all their
    sentences in 03 are not in text_level and not present in the
    result - check_table() reports them via the 'missing' function.
    """
    return text_level.merge(llm_fields, on="record_id", how="left")


def check_table(table, llm_fields):
    """Sanity checks: row count vs EXPECTED_TEXTS, empty finish_reason."""
    print(f"rows in text_level: {len(table)} (expected {EXPECTED_TEXTS})")
    if len(table) < EXPECTED_TEXTS:
        # Texts that exist in jsonl but have no row in the table = texts without suitable sentences.
        missing = set(llm_fields["record_id"]) - set(table["record_id"])
        print("record_id not in text_level:")
        for record_id in sorted(missing):
            print("  ", record_id)

    empty = table[table["finish_reason"].isna()]
    print(f"empty finish_reason: {len(empty)}")
    for record_id in empty["record_id"]:
        print("  ", record_id)


def main():
    """Build text_level.csv from metrics.csv and the raw JSONL outputs."""
    sentences = read_sentence_metrics()
    text_level = make_text_level(sentences)
    llm_fields = read_llm_fields()
    table = join_tables(text_level, llm_fields)

    check_table(table, llm_fields)

    table.to_csv(METRICS_DIR / "text_level.csv", index=False)


if __name__ == "__main__":
    main()
