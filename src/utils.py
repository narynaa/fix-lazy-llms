"""Shared utilities for the Fixing Lazy LLMs experiments."""

import json
import os
import re
import time
import random
import hashlib
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import numpy as np
from openai import OpenAI

load_dotenv()

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# API setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cache for API responses to avoid redundant calls
CACHE_DIR = PROJECT_ROOT / "results" / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_key(model: str, messages: list, temperature: float) -> str:
    """Generate a deterministic cache key for an API call."""
    content = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature},
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def call_llm(
    messages,
    model,
    temperature=0.0,
    max_tokens=1024,
):
    client = OpenAI(
        base_url=os.environ.get("GENERATOR_BASE_URL"),
        api_key=os.environ.get("GENERATOR_API_KEY", "dummy"),
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return {
        "content": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def load_gsm8k(split: str = "test", n: Optional[int] = None) -> list[dict]:
    """Load GSM8K dataset. Returns list of {question, answer, numerical_answer}."""
    path = DATASETS_DIR / "gsm8k" / f"{split}.jsonl"
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            # Extract the numerical answer after ####
            num_answer = item["answer"].split("####")[-1].strip()
            # Remove commas from numbers like 70,000
            num_answer = num_answer.replace(",", "")
            data.append(
                {
                    "question": item["question"],
                    "full_answer": item["answer"],
                    "numerical_answer": num_answer,
                }
            )
    if n is not None:
        random.seed(SEED)
        data = random.sample(data, min(n, len(data)))
    return data


def extract_numerical_answer(response: str) -> Optional[str]:
    """Extract the final numerical answer from an LLM response.

    Looks for patterns like "#### 42", "the answer is 42", "= 42", etc.
    Returns the number as a string, or None if not found.
    """
    # Try #### pattern first (most explicit)
    match = re.search(r"####\s*([\-\d,\.]+)", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Try "the answer is X" pattern
    match = re.search(
        r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*\$?([\-\d,\.]+)",
        response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "").strip()

    # Try "= $X" or "= X" at end of lines
    matches = re.findall(r"=\s*\$?([\-\d,\.]+)\s*$", response, re.MULTILINE)
    if matches:
        return matches[-1].replace(",", "").strip()

    # Try boxed answer (LaTeX style)
    match = re.search(r"\\boxed\{([\-\d,\.]+)\}", response)
    if match:
        return match.group(1).replace(",", "").strip()

    # Last resort: find the last number in the response
    matches = re.findall(r"\b([\-]?\d[\d,]*\.?\d*)\b", response)
    if matches:
        return matches[-1].replace(",", "").strip()

    return None


def check_answer(predicted: Optional[str], gold: str) -> bool:
    """Check if predicted answer matches gold answer numerically."""
    if predicted is None:
        return False
    try:
        pred_val = float(predicted.replace(",", ""))
        gold_val = float(gold.replace(",", ""))
        return abs(pred_val - gold_val) < 1e-6
    except ValueError:
        return predicted.strip() == gold.strip()


def count_tokens_approx(text: str) -> int:
    """Approximate token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)
