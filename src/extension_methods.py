from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re

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


def _split_think_final(text: str) -> Tuple[str, str]:
    """
    Parse:
      THINK:
      ...
      END_THINK
      FINAL: ...
    Return (think, final_text)
    """
    t = (text or "").strip()
    think = ""
    final = ""

    m = re.search(r"(?is)think\s*:\s*(.*?)\s*end_think", t)
    if m:
        think = m.group(1).strip()

    mf = re.search(r"(?im)^\s*final\s*:\s*(.+?)\s*$", t)
    if mf:
        final = mf.group(1).strip()
    else:
        # fallback: last non-empty line
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        final = lines[-1] if lines else ""

    return think, final


def s1_simple(prompt: str, cfg: RunConfig, rounds: int = 2):
    msgs = [
        {"role": "system", "content": "You are a careful assistant."},
        {
            "role": "user",
            "content": prompt
            + "\nThink step by step and give FINAL: <answer>.",
        },
    ]

    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for i in range(rounds):
        out = call_llm(
            msgs,
            model=cfg.model,
            temperature=0.0,
            max_tokens=cfg.max_tokens,
        )

        content = out["content"]

        for k in total_usage:
            total_usage[k] += out["usage"][k]

        msgs.append({"role": "assistant", "content": content})

        if i < rounds - 1:
            msgs.append(
                {
                    "role": "user",
                    "content": "Double-check your reasoning before finalizing.",
                }
            )

    return {"content": content, "usage": total_usage}
