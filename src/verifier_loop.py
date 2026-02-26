from utils import call_llm


def verifier_reprompt(
    base_prompt,
    cfg,
    verifier_fn,
    *,
    max_attempts=3,
    include_prev=True,
):
    total_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    prev_answer = None
    feedback = None

    attempts = []

    for attempt in range(max_attempts):

        if attempt == 0:
            prompt = base_prompt
        else:
            if include_prev and prev_answer:
                prompt = f"""{base_prompt}

Your previous answer:
{prev_answer}

Feedback:
{feedback}

Your previous answer was incorrect. Try again. Output FINAL: <answer>.
"""
            else:
                prompt = f"""{base_prompt}

Feedback:
{feedback}

Try again. Output FINAL: <answer>.
"""

        out = call_llm(
            [
                {"role": "system", "content": "You are a careful assistant."},
                {"role": "user", "content": prompt},
            ],
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )

        answer = out["content"]

        for k in total_usage:
            total_usage[k] += out["usage"][k]

        passed, feedback = verifier_fn(answer)

        attempts.append(
            {
                "attempt": attempt + 1,
                "passed": passed,
                "content": answer,
                "feedback": feedback,
            }
        )

        if passed:
            return {
                "content": answer,
                "usage": total_usage,
                "meta": {"attempts": attempts, "passed": True},
            }

        prev_answer = answer

    return {
        "content": prev_answer,
        "usage": total_usage,
        "meta": {"attempts": attempts, "passed": False},
    }
