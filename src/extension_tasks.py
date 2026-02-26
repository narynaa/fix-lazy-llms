# extension_tasks.py

import re
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset

SEED = 42
random.seed(SEED)


@dataclass
class Example:
    id: str
    prompt: str
    gold: Any
    meta: Dict[str, Any]


class Task:
    name: str

    def load(self, n: int) -> List[Example]:
        raise NotImplementedError

    def rule_verify(self, pred: str, ex: Example) -> bool:
        return True

    def score(self, pred: str, ex: Example) -> bool:
        raise NotImplementedError


# ---------- helpers ----------


import re


def extract_final_answer(text: str) -> str:
    """
    Extract the model's final answer robustly.
    - Prefer 'FINAL: ...'
    - Ignore trailing CONFIDENCE lines or extra commentary
    - Fallback: last non-empty line
    """
    t = (text or "").strip()

    # Prefer explicit FINAL:
    m = re.search(r"(?im)^\s*final\s*:\s*(.+?)\s*$", t)
    if m:
        return m.group(1).strip()

    # If the model used "Final answer:"
    m = re.search(r"(?im)^\s*final\s+answer\s*:\s*(.+?)\s*$", t)
    if m:
        return m.group(1).strip()

    # If there is a CONFIDENCE line at end, remove it
    t = re.sub(r"(?im)^\s*confidence\s*:\s*\d+\s*$", "", t).strip()

    # Fallback: last non-empty line (after removing confidence)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def extract_number(text: str) -> Optional[str]:
    nums = re.findall(r"\b([\-]?\d+\.?\d*)\b", text)
    return nums[-1] if nums else None


def numeric_match(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except:
        return pred.strip() == gold.strip()


# ---------- GSM8K ----------


class GSM8KTask(Task):
    name = "gsm8k"

    def load(self, n: int):
        ds = load_dataset("gsm8k", "main", split="test")
        idxs = random.sample(range(len(ds)), n)
        return [
            Example(
                id=f"gsm8k-{i}",
                prompt=ds[i]["question"],
                gold=ds[i]["answer"].split("####")[-1].strip(),
                meta={},
            )
            for i in idxs
        ]

    def rule_verify(self, pred, ex):
        return extract_number(pred) is not None

    def score(self, pred, ex):
        return numeric_match(extract_number(pred), ex.gold)


# ---------- SVAMP ----------


class SVAMPTask(Task):
    name = "svamp"

    def load(self, n: int):
        ds = load_dataset("ChilleD/SVAMP", split="test")
        idxs = random.sample(range(len(ds)), n)
        return [
            Example(
                id=f"svamp-{i}",
                prompt=ds[i]["Body"] + "\n" + ds[i]["Question"],
                gold=str(ds[i]["Answer"]),
                meta={},
            )
            for i in idxs
        ]

    def rule_verify(self, pred, ex):
        return extract_number(pred) is not None

    def score(self, pred, ex):
        norm_pred = pred.replace(",", "")
        norm_gold = ex.gold.replace(",", "")
        return numeric_match(extract_number(norm_pred), norm_gold)


# ---------- TruthfulQA ----------


class TruthfulQATask(Task):
    name = "truthfulqa"

    def load(self, n: int):
        ds = load_dataset("truthful_qa", "generation", split="validation")
        idxs = random.sample(range(len(ds)), n)
        return [
            Example(
                id=f"truthfulqa-{i}",
                prompt=ds[i]["question"],
                gold=ds[i]["correct_answers"],
                meta={"incorrect": ds[i]["incorrect_answers"]},
            )
            for i in idxs
        ]

    def normalize(self, s):
        s = s.lower()
        s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
        return s.strip()

    def score(self, pred, ex):
        p = self.normalize(pred)

        for ans in ex.gold:
            g = self.normalize(ans)
            if (g in p) or (p in g):
                return True

        return False


# ---------- DROP ----------


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    # remove leading labels
    s = re.sub(r"^\s*final\s*:\s*", "", s).strip()
    # strip trailing punctuation
    s = re.sub(r"[^\w\s\.\-%/]", "", s)
    return s


def _extract_numberish(s: str):
    # captures ints, floats, percents
    m = re.findall(r"[-]?\d+(?:\.\d+)?%?", s.replace(",", ""))
    return m[-1] if m else None


class DROPTask(Task):
    name = "drop"

    def load(self, n: int):
        ds = load_dataset("drop", split="validation")
        idxs = random.sample(range(len(ds)), n)
        return [
            Example(
                id=f"drop-{i}",
                prompt=f"{ds[i]['passage']}\n\n{ds[i]['question']}",
                gold=ds[i],
                meta={
                    "passage": ds[i]["passage"],
                    "question": ds[i]["question"],
                    "drop_idx": int(i),
                },
            )
            for i in idxs
        ]

    def rule_verify(self, pred, ex):
        return (
            extract_number(pred) or pred.lower() in ex.meta["passage"].lower()
        )

    def score(self, pred: str, ex: Example) -> bool:
        pred_n = _norm(pred)

        golds = []

        # spans
        spans = ex.gold.get("answers_spans", {}).get("spans", [])
        golds.extend(spans)

        # numbers
        nums = ex.gold.get("answers_numbers", {}).get("numbers", [])
        golds.extend([str(x) for x in nums])

        # date
        date = ex.gold.get("answers_date", {}).get("date", {})
        if date:
            # e.g. "march 3 1999" (loose)
            parts = [
                date.get("month", ""),
                date.get("day", ""),
                date.get("year", ""),
            ]
            golds.append(" ".join([p for p in parts if p]).strip())

        golds_n = [_norm(g) for g in golds if g]

        # direct exact match
        if any(pred_n == g for g in golds_n):
            return True

        # allow containment (handles "7 aircraft" vs "7")
        if any(pred_n in g or g in pred_n for g in golds_n):
            return True

        # numeric fallback (handles percent/commas)
        pn = _extract_numberish(pred_n)
        if pn is not None:
            for g in golds_n:
                gn = _extract_numberish(g)
                if gn is not None and pn == gn:
                    return True

        return False


# ---------- HumanEval ----------


class HumanEvalTask(Task):
    name = "humaneval"

    def load(self, n: int):
        ds = load_dataset("openai_humaneval", split="test")
        idxs = random.sample(range(len(ds)), n)
        out = []
        for i in idxs:
            item = ds[i]
            out.append(
                Example(
                    id=item["task_id"],
                    prompt=item["prompt"],
                    gold={
                        "test": item["test"],
                        "entry_point": item["entry_point"],
                    },
                    meta={},
                )
            )
        return out


# ---------- registry ----------


def get_task(name: str) -> Task:
    if name == "gsm8k":
        return GSM8KTask()
    if name == "truthfulqa":
        return TruthfulQATask()
    if name == "svamp":
        return SVAMPTask()
    if name == "drop":
        return DROPTask()
    if name == "humaneval":
        return HumanEvalTask()
    raise ValueError(name)
