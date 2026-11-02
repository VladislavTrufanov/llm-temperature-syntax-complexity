import os
import sys
import json
import uuid
import time
from datetime import datetime
from itertools import product

from dotenv import load_dotenv
from groq import Groq

from common import PROJECT_ROOT, PROMPTS_DIR, LLM_OUTPUT_DIR, dataset_stem, load_grid

# Sampling and output shape.
TOP_P = 1.0
MAX_TOKENS = 384  # Limits output length (not a rate-limit knob).

# Rate limiting and retry settings for Groq API.
PAUSE_SECONDS = 30 # Pause between requests to avoid rate limiting.
MAX_RETRIES = 3 # Total attempts per request, including the first one.
RETRY_WAIT = 30 # Backoff time in seconds between retries.


def groq_client():
    """Return a Groq client instance using the API key from the .env file."""
    load_dotenv(PROJECT_ROOT / '.env')
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        print('GROQ_API_KEY not found in .env')
        sys.exit(1)
    return Groq(api_key=api_key)


def llm_reasoning_effort(model):
    kwargs = {}
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = "low"
    elif "qwen3.8" in model:
        kwargs["reasoning_effort"] = "default"
    return kwargs


def generate_texts(client, model, messages, temperature, seed):
    """Request a single completion from Groq and normalize the response.

    Returns a dict with the output, api_* metadata, token counts,
    finish_reason, and error (None on success).
    """
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=TOP_P,
        seed=seed,
        max_tokens=MAX_TOKENS,
        **llm_reasoning_effort(model),
    )
    return {
        "output": completion.choices[0].message.content,
        "api_id": completion.id,
        "api_model": completion.model,
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "total_tokens": completion.usage.total_tokens,
        "finish_reason": completion.choices[0].finish_reason,
        "error": None,
    }


def error_record(error):
    """Return the same keys as generate_texts, but with error information.
    
    All metrics are None; error holds "ClassName: message". This keeps
    the JSONL schema identical for successful and failed records.
    """
    return {
        "output": None,
        "api_id": None,
        "api_model": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "error": f"{type(error).__name__}: {error}",
    }


def fetch_data_with_retries(client, model, messages, temperature, seed):
    """Call generate_texts with retries.

    Makes up to MAX_RETRIES attempts in total, waiting RETRY_WAIT seconds
    between them (constants at the top of the file). When all attempts
    fail, returns error_record(e) instead of raising.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return generate_texts(client, model, messages, temperature, seed)
        except Exception as e:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)
            if attempt == MAX_RETRIES:
                return error_record(e)
            time.sleep(RETRY_WAIT)


def append_record(path, record):
    """Append a record to a JSONL file at the specified path."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    """Iterate over the full grid and write one JSONL file per
    (model, language, prompt_type) triple into LLM_OUTPUT_DIR.
    """
    LLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = groq_client()
    grid = load_grid()

    combinations = product(
        grid["models"], 
        grid["languages"], 
        grid["prompt_types"],
        grid["seeds"], 
        grid["temperatures"],
    )
    for i, (model, language, prompt_type, seed, temperature) in enumerate(combinations, start=1):
        print(f"{i}: {model} {language} {prompt_type} temp={temperature}", flush=True)

        prompt_text = (PROMPTS_DIR / f"{language}_{prompt_type}.txt").read_text(encoding="utf-8")
        record = {
            "record_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "seed": seed,
            "temperature": temperature,
            "language": language,
            "prompt_type": prompt_type,
            "prompt": prompt_text,
            "max_tokens": MAX_TOKENS,
        }

        messages = [{"role": "user", "content": prompt_text}]
        record.update(fetch_data_with_retries(client, model, messages, temperature, seed))

        path = LLM_OUTPUT_DIR / f"{dataset_stem(model, language, prompt_type)}.jsonl"
        append_record(path, record)

        time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()