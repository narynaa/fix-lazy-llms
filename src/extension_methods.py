from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from utils import call_llm
from extension_tasks import extract_final_answer

SYSTEM_BASE = "You are a careful assistant. Follow instructions exactly."

CHECKLIST_TEMPLATES = {
    "math": (
        "Use this checklist:\n"
        "1) Extract givens\n"
        "2) Plan the solution\n"
        "3) Compute carefully\n"
        "4) Sanity-check magnitude/units\n"
        "Then give FINAL: <answer>.\n"
    ),
    "drop": (
        "Use this checklist:\n"
        "1) Find relevant sentence(s) in the passage\n"
        "2) Extract needed entities/numbers\n"
        "3) Decide operation (count/add/compare)\n"
        "4) Answer using words from passage or a number\n"
        "Then give FINAL: <answer>.\n"
    ),
    "truthfulqa": (
        "Use this checklist:\n"
        "1) Restate what is being asked\n"
        "2) Avoid common misconceptions\n"
        "3) If unsure, say you are unsure rather than guessing\n"
        "Then give FINAL: <answer>.\n"
    ),
    "humaneval": (
        "Use this checklist:\n"
        "1) Understand function signature + constraints\n"
        "2) Consider edge cases\n"
        "3) Write clear, correct Python\n"
        "Return ONLY code.\n"
    ),
}


@dataclass
class RunConfig:
    model: str
    temperature: float = 0.0
    max_tokens: int = 2048


def baseline(
    prompt: str, cfg: RunConfig, system: str = SYSTEM_BASE
) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    out = call_llm(
        msgs,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return {"content": out["content"], "usage": out["usage"]}


def checklist(
    prompt: str, checklist_kind: str, cfg: RunConfig
) -> Dict[str, Any]:
    system = SYSTEM_BASE + "\n" + CHECKLIST_TEMPLATES[checklist_kind]
    return baseline(prompt, cfg, system=system)


def conditional_critique(
    prompt: str, cfg: RunConfig, critique_prompt: str, threshold: int = 70
) -> Dict[str, Any]:
    """
    1) Generate initial answer + confidence (0-100)
    2) If confidence < threshold, run a critique pass; else keep initial.
    """
    initial_msgs = [
        {
            "role": "system",
            "content": SYSTEM_BASE
            + "\nReturn exactly two lines:\nFINAL: <answer>\nCONFIDENCE: <0-100>",
        },
        {"role": "user", "content": prompt},
    ]
    init = call_llm(
        initial_msgs,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    text = init["content"]
    # parse confidence
    conf = None
    import re

    m = re.search(r"confidence\s*[:=]\s*(\d+)", text, re.I)
    if m:
        conf = int(m.group(1))
    # default: be conservative and critique
    do_crit = (conf is None) or (conf < threshold)

    if not do_crit:
        return {
            "content": text,
            "usage": init["usage"],
            "meta": {"confidence": conf, "critiqued": False},
        }

    critique_msgs = [
        {
            "role": "system",
            "content": SYSTEM_BASE + "\nYou must follow formatting exactly.",
        },
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": text},
        {
            "role": "user",
            "content": "Check the solution carefully. If the FINAL answer is wrong, fix it. "
            "If the FINAL answer is correct, repeat it unchanged.\n"
            "Return exactly one line:\nFINAL: <answer>",
        },
    ]
    crit = call_llm(
        critique_msgs,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    usage = {
        "prompt_tokens": init["usage"]["prompt_tokens"]
        + crit["usage"]["prompt_tokens"],
        "completion_tokens": init["usage"]["completion_tokens"]
        + crit["usage"]["completion_tokens"],
        "total_tokens": init["usage"]["total_tokens"]
        + crit["usage"]["total_tokens"],
    }
    return {
        "content": crit["content"],
        "usage": usage,
        "meta": {"confidence": conf, "critiqued": True},
    }


def verifier_selection(
    prompt: str,
    cfg: RunConfig,
    rule_verify_fn,
    k: int = 3,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Generate k candidates (diversity) and select first that passes rule_verify_fn(candidate).
    This verifier does NOT use gold labels.
    """
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    candidates: List[str] = []

    for _ in range(k):
        out = call_llm(
            [
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": prompt},
            ],
            model=cfg.model,
            temperature=temperature,
            max_tokens=cfg.max_tokens,
            use_cache=False,  # need diversity
        )
        candidates.append(out["content"])
        for key in total:
            total[key] += out["usage"][key]

    chosen = candidates[0]
    for c in candidates:
        if rule_verify_fn(c):
            chosen = c
            break

    return {
        "content": chosen,
        "usage": total,
        "meta": {"k": k, "passed": rule_verify_fn(chosen)},
    }
