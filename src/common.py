import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
PROMPTS_DIR = CONFIGS_DIR / "prompts"

RESULTS_DIR = PROJECT_ROOT / "results"
LLM_OUTPUT_DIR = RESULTS_DIR / "llm_output"
PARSED_DIR = RESULTS_DIR / "parsed"
METRICS_DIR = RESULTS_DIR / "metrics"
ANALYSIS_DIR = RESULTS_DIR / "analysis"

LANG_CODES = {"english": "en", "french": "fr"}

METADATA_FIELDS = ("model", "language", "prompt_type", "temperature", "seed", "record_id")

def load_grid():
    with open(CONFIGS_DIR / "grid.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_name(text):
    return re.sub(r"[^\w\-.]", "_", text)

def dataset_stem(model, language, prompt_type):
    return f"{safe_name(model)}_{language}_{prompt_type}"