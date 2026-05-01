"""
Milestone 1 verification script.

Checks:
  1. Ollama server is running and all-minilm:l6-v2 is available
  2. LongMemEval dataset is present and has the expected structure
  3. Split file is present with the expected dev/held_out counts

Usage:
    .venv/bin/python benchmarks/verify_milestone1.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "benchmarks/data/longmemeval/longmemeval_s_cleaned.json"
SPLIT   = ROOT / "benchmarks/data/lme_split_50_450.json"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
INFO = "\033[34mℹ\033[0m"

errors: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    icon = PASS if ok else FAIL
    print(f"  {icon}  {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        errors.append(label)


# ── 1. Ollama ──────────────────────────────────────────────────────────────

print("\n[1] Ollama server")

ollama_up = False
try:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
        tags = json.loads(r.read())
    models = [m["name"] for m in tags.get("models", [])]
    ollama_up = True
    check("Ollama server is running", True)
    minilm_present = any("all-minilm" in m for m in models)
    check(
        "all-minilm:l6-v2 is pulled",
        minilm_present,
        f"found: {[m for m in models if 'minilm' in m] or 'none'}",
    )
    if not minilm_present:
        print(f"  {INFO}  Run:  ollama pull all-minilm:l6-v2")
except urllib.error.URLError:
    check("Ollama server is running", False, "http://localhost:11434 unreachable")
    check("all-minilm:l6-v2 is pulled", False, "skipped — server not running")
    print(f"  {INFO}  Start Ollama and re-run this script.")

# ── 2. Dataset ─────────────────────────────────────────────────────────────

print("\n[2] LongMemEval dataset")

if not DATASET.exists():
    check("Dataset file present", False, str(DATASET))
    print(f"  {INFO}  Run:")
    print(f"         mkdir -p {DATASET.parent}")
    print(f'         curl -L -o {DATASET} \\')
    print('           "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"')
else:
    size_mb = DATASET.stat().st_size / 1_048_576
    check("Dataset file present", True, f"{size_mb:.0f} MB")

    with open(DATASET) as f:
        data: list[dict] = json.load(f)

    check("500 questions loaded", len(data) == 500, f"got {len(data)}")

    # field presence
    sample = data[0]
    for field in ("question_id", "question", "answer", "question_type",
                  "question_date", "haystack_sessions", "haystack_session_ids",
                  "haystack_dates", "answer_session_ids"):
        check(f"field '{field}' present", field in sample)

    # parallel list lengths
    if all(f in sample for f in ("haystack_sessions", "haystack_session_ids", "haystack_dates")):
        ns = len(sample["haystack_sessions"])
        ni = len(sample["haystack_session_ids"])
        nd = len(sample["haystack_dates"])
        check(
            "haystack_sessions / _ids / _dates are parallel",
            ns == ni == nd,
            f"sessions={ns} ids={ni} dates={nd}",
        )

    # question type distribution
    types = Counter(e["question_type"] for e in data)
    expected = {
        "multi-session": 133,
        "temporal-reasoning": 133,
        "knowledge-update": 78,
        "single-session-user": 70,
        "single-session-assistant": 56,
        "single-session-preference": 30,
    }
    print(f"\n  Question type breakdown:")
    for qtype, expected_n in expected.items():
        actual_n = types.get(qtype, 0)
        icon = PASS if actual_n == expected_n else FAIL
        print(f"    {icon}  {qtype}: {actual_n}  (expected {expected_n})")
        if actual_n != expected_n:
            errors.append(f"question type count mismatch: {qtype}")

    # avg sessions per question
    avg_sessions = sum(len(e["haystack_sessions"]) for e in data) / len(data)
    check("avg ~53 sessions per question", 45 <= avg_sessions <= 60, f"{avg_sessions:.1f}")

# ── 3. Split file ──────────────────────────────────────────────────────────

print("\n[3] Split file")

if not SPLIT.exists():
    check("Split file present", False, str(SPLIT))
    print(f"  {INFO}  Copy from mempalace:")
    print(f"         cp /path/to/mempalace/benchmarks/lme_split_50_450.json {SPLIT}")
else:
    check("Split file present", True)
    with open(SPLIT) as f:
        split = json.load(f)

    check("seed is 42", split.get("seed") == 42, f"got {split.get('seed')}")
    check("dev split has 50 IDs", len(split.get("dev", [])) == 50, f"got {len(split.get('dev', []))}")
    check("held_out split has 450 IDs", len(split.get("held_out", [])) == 450, f"got {len(split.get('held_out', []))}")

    # no overlap
    dev_set = set(split.get("dev", []))
    held_set = set(split.get("held_out", []))
    check("no overlap between dev and held_out", len(dev_set & held_set) == 0,
          f"{len(dev_set & held_set)} overlapping IDs")

    # all IDs exist in dataset (only if dataset loaded)
    if DATASET.exists():
        with open(DATASET) as f:
            data = json.load(f)
        all_ids = {e["question_id"] for e in data}
        missing_dev = dev_set - all_ids
        missing_held = held_set - all_ids
        check("all dev IDs in dataset", len(missing_dev) == 0, f"{len(missing_dev)} missing")
        check("all held_out IDs in dataset", len(missing_held) == 0, f"{len(missing_held)} missing")

# ── Summary ────────────────────────────────────────────────────────────────

print()
if errors:
    print(f"\033[31m✗ Milestone 1 incomplete — {len(errors)} issue(s):\033[0m")
    for e in errors:
        print(f"  • {e}")
    sys.exit(1)
else:
    print(f"\033[32m✓ Milestone 1 complete — all checks passed.\033[0m")
