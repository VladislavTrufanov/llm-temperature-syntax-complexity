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

TOP_P = 1.0
MAX_TOKENS = 384
PAUSE_SECONDS = 30
MAX_RETRIES = 3
RETRY_WAIT = 30

def groq_client():
    load_dotenv(PROJECT_ROOT / '.env')
    api_key = os.getenv("GROQ_API_KEY")
    if api_key is None:
        print('Add API Key')
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
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return generate_texts(client, model, messages, temperature, seed)
        except Exception as e:
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)
            if attempt == MAX_RETRIES:
                return error_record(e)
            time.sleep(RETRY_WAIT)

def append_record(path, record):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

def main():
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