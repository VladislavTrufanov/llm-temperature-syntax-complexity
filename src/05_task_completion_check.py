import json
import re

import pandas as pd

from common import LLM_OUTPUT_DIR, METRICS_DIR

# The four hours from the prompt: 8:00, 22:00 (weekdays) and 10:00, 18:00 (weekend)
HOURS = [8, 22, 10, 18]

# Fixed seed, so the "10 random texts" are the same on every run
RANDOM_SEED = 12


def normalize_spaces(text):
    """Replace narrow or no-break spaces (U+202F, U+00A0) with a normal space."""
    return text.replace("\u202f", " ").replace("\u00a0", " ")


# 12-hour English time: "8 AM", "8am", "8:00 AM", "10:00 p.m.", "06:00 pm", "10PM" ...
EN_TIME_RE = re.compile(
    r"(?<!\d)(\d{1,2})(?::\d{2})?\s*([ap])\.?\s*m\.?(?!\w)",
    re.IGNORECASE,
)


def replace_english_times(text):
    """Convert 12-hour notation to 24-hour, so the plain digit check below works."""
    def to_24h(match):
        hour = int(match.group(1))
        meridiem = match.group(2).lower()
        if meridiem == "p" and hour < 12:
            hour += 12
        elif meridiem == "a" and hour == 12:
            hour = 0
        return str(hour)

    return EN_TIME_RE.sub(to_24h, text)


def has_hour(text, hour):
    """r'(?<!\d) and (?!\d)' mean hour must stand alone, not inside a longer number."""
    pattern = rf"(?<!\d)0?{hour}(?!\d)"
    return re.search(pattern, text) is not None


def check_record(record):
    """One text = one row of the result table."""
    text = normalize_spaces(record["output"])
    if record["language"] == "english":
        text = replace_english_times(text)

    row = {"record_id": record["record_id"]}
    for hour in HOURS:
        row[f"has_{hour}"] = has_hour(text, hour)

    # True counts as 1, False as 0, so sum() counts the found hours.
    row["n_hours_found"] = sum(row[f"has_{hour}"] for hour in HOURS)
    row["task_ok"] = row["n_hours_found"] == len(HOURS)
    return row


def check_all_files():
    rows = []
    for path in sorted(LLM_OUTPUT_DIR.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                rows.append(check_record(record))
    return pd.DataFrame(rows)


def print_rows(rows):
    for _, row in rows.iterrows():
        print(row["record_id"][:8], row["n_hours_found"])


def print_report(table):
    # ~ means "not": here, the rows where task_ok is False
    print("task_ok = False:")
    print_rows(table[~table["task_ok"]])

    ok_rows = table[table["task_ok"]]
    print("10 random task_ok = True:")
    print_rows(ok_rows.sample(n=min(10, len(ok_rows)), random_state=RANDOM_SEED))


def main():
    table = check_all_files()
    table.to_csv(METRICS_DIR / "task_ok.csv", index=False)
    print_report(table)


if __name__ == "__main__":
    main()