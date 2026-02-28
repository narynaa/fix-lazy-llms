import argparse
import re
import json
from pathlib import Path
from typing import Dict, Any, List

from utils import RESULTS_DIR
from extension_tasks import get_task, HumanEvalTask, extract_final_answer
from extension_methods import (
    RunConfig,
    baseline,
    checklist,
    conditional_critique,
    s1_simple,
)

# HumanEval runner
from human_eval.evaluation import evaluate_functional_correctness
from verifier_loop import verifier_reprompt
from rm_verifier import rm_verify_drop, rm_verify_svamp

CRITIQUE_GENERIC = (
    "Carefully review the answer above. If there are mistakes, correct them. "
    "Do not change answers unless you are confident there is an error."
)


def run_humaneval_eval(
    samples: List[Dict[str, Any]],
    out_dir: Path,
    problems_subset: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_file = out_dir / "humaneval_samples.jsonl"
    with sample_file.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Write subset problems file (only the attempted problems)
    problem_file = out_dir / "humaneval_problems_subset.jsonl"
    with problem_file.open("w") as f:
        for p in problems_subset:
            f.write(json.dumps(p) + "\n")

    res = evaluate_functional_correctness(
        sample_file=str(sample_file),
        k=[1],
        n_workers=4,
        timeout=3.0,
        problem_file=str(problem_file),  # <-- IMPORTANT
    )
    return res


def extract_code_only(text: str) -> str:
    t = (text or "").strip()

    # Prefer fenced code
    m = re.search(
        r"```(?:python)?\s*(.*?)\s*```", t, flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    # Otherwise, drop any FINAL/CONFIDENCE lines if present
    t = re.sub(r"(?im)^\s*final\s*:.*$", "", t).strip()
    t = re.sub(r"(?im)^\s*confidence\s*:.*$", "", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks", type=str, default="gsm8k,truthfulqa,svamp,drop,humaneval,math500"
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="baseline,checklist,conditional_critique,verifier",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=50,
        help="examples per dataset (small for first run)",
    )
    ap.add_argument("--model", type=str, default="gpt-4.1-mini")
    ap.add_argument("--out", type=str, default="extension_run")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    cfg = RunConfig(model=args.model)
    out_name = args.model.replace(".", "_")
    out_name = out_name.replace("/", "")

    out_path = RESULTS_DIR / f"{out_name}.json"
    results: Dict[str, Any] = {"config": vars(args), "results": {}}

    for task_name in tasks:
        task = get_task(task_name)
        examples = task.load(args.n)
        results["results"][task_name] = {}

        for method in methods:
            print(f"\n== Task={task_name} Method={method} n={len(examples)} ==")
            rows = []
            humaneval_samples = []
            humaneval_problems_subset = []

            for ex in examples:
                # Build method-specific prompt
                if task_name in ("gsm8k", "svamp"):
                    base_prompt = f"Solve step by step. End with FINAL: <number>.\n\n{ex.prompt}"
                    checklist_kind = "math"
                elif task_name == "drop":
                    base_prompt = f"Answer using the passage. End with FINAL: <answer>.\n\n{ex.prompt}"
                    checklist_kind = "drop"
                elif task_name == "truthfulqa":
                    base_prompt = f"Answer the question. End with FINAL: <answer>.\n\n{ex.prompt}"
                    checklist_kind = "truthfulqa"
                elif task_name == "math500":
                    base_prompt = f"Solve step by step. End with FINAL: <answer>.\n\n{ex.prompt}"
                    checklist_kind = "math"
                elif task_name == "humaneval":
                    base_prompt = f"{ex.prompt}\n# Write the solution in Python. Return ONLY code.\n"
                    checklist_kind = "humaneval"
                else:
                    base_prompt = ex.prompt
                    checklist_kind = "truthfulqa"

                # Run
                if method == "baseline":
                    out = baseline(base_prompt, cfg)
                elif method == "checklist":
                    out = checklist(base_prompt, checklist_kind, cfg)
                elif method == "conditional_critique":
                    out = conditional_critique(
                        base_prompt, cfg, CRITIQUE_GENERIC, threshold=70
                    )
                elif method == "s1_budget_forcing":
                    out = s1_simple(
                        base_prompt,
                        cfg,
                    )
                elif method == "verifier_reprompt_rm":
                    if task_name == "drop":

                        def verifier_fn(ans, ex=ex):
                            return rm_verify_drop(
                                answer=ans,
                                passage=ex.meta["passage"],
                                question=ex.meta["question"],
                            )

                        out = verifier_reprompt(
                            base_prompt,
                            cfg,
                            verifier_fn,
                            max_attempts=3,
                            include_prev=False,
                        )
                    elif task_name == "svamp":

                        def verifier_fn(ans):
                            return rm_verify_svamp(
                                prompt=base_prompt,
                                answer=ans,
                            )

                        out = verifier_reprompt(
                            base_prompt,
                            cfg,
                            verifier_fn,
                            max_attempts=3,
                            include_prev=False,
                        )
                else:
                    raise ValueError(f"Unknown method: {method}")

                content = out["content"]
                pred_final = (
                    content
                    if task_name == "humaneval"
                    else extract_final_answer(content)
                )

                row = {
                    "id": ex.id,
                    "pred": pred_final,
                    "usage": out["usage"],
                    "meta": out.get("meta", {}),
                }

                if task_name == "humaneval":
                    clean = extract_code_only(content)
                    humaneval_samples.append(
                        {"task_id": ex.id, "completion": clean}
                    )
                    humaneval_problems_subset.append(
                        {
                            "task_id": ex.id,
                            "prompt": ex.prompt,
                            "test": ex.gold["test"],
                            "entry_point": ex.gold["entry_point"],
                        }
                    )
                    row["correct"] = None
                else:
                    row["correct"] = bool(task.score(pred_final, ex))

                rows.append(row)

            # Score / summarize
            summary = {}
            if task_name == "humaneval":
                eval_res = run_humaneval_eval(
                    humaneval_samples,
                    RESULTS_DIR / f"{args.out}_humaneval_eval",
                    humaneval_problems_subset,
                )
                summary["pass@1"] = eval_res.get("pass@1", None)
                summary["raw_eval"] = eval_res
            else:
                acc = sum(1 for r in rows if r["correct"]) / max(1, len(rows))
                avg_tokens = sum(
                    r["usage"]["total_tokens"] for r in rows
                ) / max(1, len(rows))
                summary = {"accuracy": acc, "avg_total_tokens": avg_tokens}

            results["results"][task_name][method] = {
                "summary": summary,
                "rows": rows,
            }

            print("Summary:", summary)

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
