import json
from pathlib import Path

MAIN_PATH = Path("first_extension_run_gsm8k_gpt-4_1-mini.json")
TEST_PATH = Path("extension_run_gpt-4_1-mini.json")
OUT_PATH = Path("first_extension_run_gsm8k_PATCHED_gpt-4_1-mini.json")

main = json.loads(MAIN_PATH.read_text())
test = json.loads(TEST_PATH.read_text())

# 1) grab "good" conditional_critique block from test
good_block = test["results"]["gsm8k"]["conditional_critique"]

# 2) replace "bugged" block in main
main["results"]["gsm8k"]["conditional_critique"] = good_block

OUT_PATH.write_text(json.dumps(main, indent=2))
print(f"Wrote patched file: {OUT_PATH}")
