import os
import yaml
import json
import uuid
import time
from groq import Groq
from pathlib import Path
from datetime import datetime
from itertools import product
from dotenv import load_dotenv

load_dotenv()

base_dir = Path(__file__).resolve().parent

project_root = base_dir.parent
configs_dir = project_root / 'configs'
prompts_dir = project_root / 'configs' / 'prompts'
output_proc = project_root / 'results' / 'llm_output' 

output_proc.mkdir(parents=True, exist_ok=True)

api_key = os.getenv("GROQ_API_KEY")
 
client = Groq(api_key=api_key)

max_tokens = 384

with open(configs_dir / "grid.yaml", 'r', encoding="utf-8") as f:
    grid = yaml.safe_load(f)

for i, (model, language, prompt_type, seed, temperature) in enumerate(
    product(
        grid["models"], grid["languages"], grid["prompt_types"],
        grid["seeds"], grid["temperatures"]
    ),
    start=1
    ):
    
    print(f"{i}: {model} {language} {prompt_type} temp={temperature}", flush=True)
    
    prompt_path = (prompts_dir / f"{language}_{prompt_type}.txt")
    prompts_text = prompt_path.read_text(encoding="utf-8")

    data = {
    "record_id": str(uuid.uuid4()),
    "timestamp": datetime.now().isoformat(),
    "model": model,
    "seed": seed,
    "temperature": temperature,
    "language": language,
    "prompt_type": prompt_type,
    "prompt": prompts_text,
    "max_tokens": max_tokens
    }

    try:
        kwargs = {}
        
        if "gpt-oss" in model:
            kwargs["reasoning_effort"] = "low"
        elif "qwen3.8" in model:
            kwargs["reasoning_effort"] = "default"
            
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompts_text
                }
            ],
            temperature=temperature,
            top_p=1,
            seed=seed,
            max_tokens=max_tokens,
            **kwargs
        )
        data["output"] = completion.choices[0].message.content
        data["api_id"] = completion.id
        data["api_model"] = completion.model
        data["prompt_tokens"] = completion.usage.prompt_tokens
        data["completion_tokens"] = completion.usage.completion_tokens
        data["total_tokens"] = completion.usage.total_tokens
        data["finish_reason"] = completion.choices[0].finish_reason
        data["error"] = None
    
    except Exception as e:
        data["output"] = None
        data["api_id"] = None
        data["api_model"] = None
        data["prompt_tokens"] = None
        data["completion_tokens"] = None
        data["total_tokens"] = None
        data["finish_reason"] = None
        data["error"] = f"{type(e).__name__}: {e}"

    line = json.dumps(data, ensure_ascii=False)

    model_name = model.replace('/', '_').replace(':', '_')
    dataset_name = f"{model_name}_{language}_{prompt_type}.jsonl"
    
    dataset_path = output_proc / dataset_name

    with open(dataset_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    
    time.sleep(30)