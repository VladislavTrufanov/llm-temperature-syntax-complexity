import pandas as pd

from common import METRICS_DIR

# The four text-level metrics made by 04_text_aggregation.py
METRICS = ["length_med", "max_depth_med", "mean_depth_med", "mdd_med"]


def read_and_join():
    """Read text_level.csv and task_ok.csv; keep only texts present in both."""
    text_level = pd.read_csv(METRICS_DIR / "text_level.csv")
    task_ok = pd.read_csv(METRICS_DIR / "task_ok.csv")
    print(f"text_level rows: {len(text_level)}")
    print(f"task_ok rows: {len(task_ok)}")
    return text_level.merge(task_ok, on="record_id", how="inner")


def make_cell_summary(table):
    """One row per cell: model x language x prompt_type x temperature."""
    aggregations = {"n_texts": ("record_id", "count")}
    for metric in METRICS:
        aggregations[f"{metric}_median"] = (metric, "median")
        aggregations[f"{metric}_min"] = (metric, "min")
        aggregations[f"{metric}_max"] = (metric, "max")
    # The mean of True/False is the share of True (True = 1, False = 0)
    aggregations["share_task_ok"] = ("task_ok", "mean")

    group = ["model", "language", "prompt_type", "temperature"]
    return table.groupby(group).agg(**aggregations).reset_index()


def make_rq3_table(table):
    """One row per model x temperature x task_ok (True/False)."""
    aggregations = {"n_texts": ("record_id", "count")}
    for metric in METRICS:
        aggregations[f"{metric}_median"] = (metric, "median")

    group = ["model", "temperature", "task_ok"]
    return table.groupby(group).agg(**aggregations).reset_index()


def main():
    table = read_and_join()
    make_cell_summary(table).to_csv(METRICS_DIR / "cell_summary.csv", index=False)
    make_rq3_table(table).to_csv(METRICS_DIR / "rq3_task_ok.csv", index=False)


if __name__ == "__main__":
    main()
