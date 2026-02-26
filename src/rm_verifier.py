from openai import OpenAI
import os
import re
import json

rm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def parse_first_json_object(s: str) -> dict:
    if not s:
        raise ValueError("Empty response")

    s = s.strip()

    # remove code fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # extract first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON found in: {s[:200]}")

    return json.loads(m.group(0))


# DROP
def rm_verify_drop(
    answer: str,
    passage: str,
    question: str,
    threshold: float = 0.6,
):
    """
    Uses GPT-4.1-mini as a reward model / judge.
    Returns (pass: bool, feedback: str)
    """

    judge_prompt = f"""
You are a strict evaluator.

PASSAGE:
{passage}

QUESTION:
{question}

MODEL ANSWER:
{answer}

Evaluate whether the answer correctly answers the question using the passage.

Return JSON:
{{
  "score": <0 to 1>,
  "verdict": "PASS" or "FAIL",
  "feedback": "short explanation"
}}
"""

    out = rm_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a strict evaluator."},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )

    text = out.choices[0].message.content

    import json

    try:
        obj = json.loads(text)
        score = obj.get("score", 0)
        verdict = obj.get("verdict", "FAIL")
        feedback = obj.get("feedback", "")
    except:
        # fallback if parsing fails
        score = 0
        verdict = "FAIL"
        feedback = text

    passed = (score >= threshold) or (verdict.upper() == "PASS")

    return passed, feedback


# SVAMP
def _extract_numberish(text: str):
    if not text:
        return None
    m = re.findall(r"[-]?\d+(?:\.\d+)?%?", text.replace(",", ""))
    return m[-1] if m else None


def rm_verify_svamp(
    prompt: str,
    answer: str,
    threshold: float = 0.6,
):
    """
    Uses GPT-4.1-mini as a judge for SVAMP-style math word problems.
    Judge re-solves and checks if the candidate answer matches.
    Returns (pass: bool, feedback: str)
    """

    judge_prompt = f"""
You are a strict evaluator for math word problems.

PROBLEM:
{prompt}

CANDIDATE ANSWER:
{answer}

Decide if the candidate answer is correct.

Guidelines:
- Solve the problem yourself (internally).
- Compare the candidate's FINAL number to the correct number.
- Be strict about numeric equality (allow commas and minor formatting differences).
- If the candidate answer is not a clear number, it should usually FAIL.

Return JSON:
{{
  "score": <0 to 1>,
  "verdict": "PASS" or "FAIL",
  "correct_answer": "<number or expression>",
  "feedback": "short explanation"
}}
"""

    out = rm_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a strict evaluator."},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0.0,
        max_tokens=250,
    )

    text = out.choices[0].message.content

    try:
        obj = json.loads(text)
        score = float(obj.get("score", 0))
        verdict = str(obj.get("verdict", "FAIL"))
        feedback = str(obj.get("feedback", ""))
        correct_answer = str(obj.get("correct_answer", ""))
    except Exception:
        score = 0.0
        verdict = "FAIL"
        feedback = text
        correct_answer = ""

    # if judge output is messy, also do a numeric match check
    cand_num = _extract_numberish(answer)
    corr_num = _extract_numberish(correct_answer)
    numeric_agree = (
        (cand_num is not None)
        and (corr_num is not None)
        and (cand_num == corr_num)
    )

    passed = (
        numeric_agree or (score >= threshold) or (verdict.upper() == "PASS")
    )
    if numeric_agree and not feedback:
        feedback = f"Numeric match: candidate={cand_num}, correct={corr_num}"

    return passed, feedback
