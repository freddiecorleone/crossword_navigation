# This cell will create a ready-to-run terminal app script for collecting crossword data.
# It reads a CSV with columns: id, clue, answer (e.g., crossword_train.csv),
# presents random clues with random filled letters, logs user responses,
# and saves rows continuously to a CSV. Run locally with:
#   python crossword_collector.py [optional_csv_path]
#
# The log will be saved as crossword_session_LOG.csv by default.

from textwrap import dedent
from pathlib import Path

script = dedent(r"""
#!/usr/bin/env python3
import csv
import json
import os
import random
import signal
import sys
import uuid
from datetime import datetime
from pathlib import Path

# ---------- Config ----------
SAVE_EVERY_N = 1   # append to file after every N trials
LOG_BASENAME = "crossword_session_LOG.csv"
RANDOM_SEED = None  # set an int for reproducibility, or None

# ---------- Helpers ----------
def normalize_answer(ans: str) -> str:
    # Uppercase and strip spaces/hyphens; keep other characters to preserve abbreviations
    return ans.upper().replace(" ", "").replace("-", "").strip()

def choose_mask_indices(L: int) -> list:
    # Uniformly choose k in {0, 1, ..., L}; then choose k unique indices to reveal
    k = random.randint(0, L)
    return sorted(random.sample(range(L), k))

def apply_mask(answer: str, idxs: list) -> str:
    chars = list(answer.upper())
    masked = []
    for i, ch in enumerate(chars):
        masked.append(ch if i in idxs else "_")
    return "".join(masked)

def count_remaining_correct(guess_norm: str, answer_norm: str, filled_idxs: list) -> int:
    # Count correctly filled letters by the user among the *unfilled* positions we showed
    # Only compare positions < min(len(guess), len(answer)) to be robust to length mismatch
    n = min(len(guess_norm), len(answer_norm))
    correct = 0
    for i in range(n):
        if i in filled_idxs:
            continue
        if guess_norm[i] == answer_norm[i]:
            correct += 1
    return correct

def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def print_rule():
    print("-"*72)
    print("Enter your answer and press <Enter>. Commands:")
    print("  :skip    -> skip this clue (records as incorrect, guess='')")
    print("  :reveal  -> show the answer, then press <Enter> to move on (records as revealed)")
    print("  :quit    -> save & exit")
    print("-"*72)
    print("Usage: python crossword_collector.py [csv_path] [log_output.csv]")
    print("       If no csv_path provided, uses crossword_data/crossword_train.csv")
    print("-"*72)

# ---------- Main ----------
def main():
    # Get the project root directory (3 levels up from the generated script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent if script_dir.name == "training_app" else script_dir.parent.parent.parent
    default_csv_path = project_root / "crossword_data" / "crossword_train.csv"
    
    # Use command line argument if provided, otherwise use default path
    if len(sys.argv) >= 2:
        src_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) >= 3 else LOG_BASENAME
    else:
        src_path = str(default_csv_path)
        out_path = LOG_BASENAME
        print(f"No CSV path provided, using default: {src_path}")

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    if not os.path.exists(src_path):
        print(f"Error: source CSV not found: {src_path}")
        sys.exit(1)

    # Load dataset
    rows = []
    with open(src_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"id", "clue", "answer"}
        if not expected.issubset(set(reader.fieldnames or [])):
            print(f"Error: CSV must have columns {expected}, got {reader.fieldnames}")
            sys.exit(1)
        for r in reader:
            if not r.get("id") or not r.get("clue") or not r.get("answer"):
                continue
            ans = r["answer"]
            if not isinstance(ans, str) or len(ans.strip()) == 0:
                continue
            rows.append({"id": r["id"], "clue": r["clue"], "answer": r["answer"]})

    if not rows:
        print("No usable rows found in the source CSV.")
        sys.exit(1)

    # Prepare logging
    session_id = str(uuid.uuid4())
    fieldnames = [
        "timestamp_utc",
        "session_id",
        "source_id",
        "clue",
        "answer_raw",
        "answer_norm",
        "answer_len",
        "clue_len",
        "filled_indices",          # JSON list of indices we pre-filled
        "filled_pairs",            # JSON list of [idx, letter] we showed
        "shown_pattern",           # String with underscores for unknowns
        "user_guess_raw",
        "user_guess_norm",
        "user_correct",            # 0/1
        "revealed",                # 0/1 (user asked to reveal)
        "num_remaining_correct"    # among positions not pre-filled
    ]

    # If the file doesn't exist, create with header
    new_file = not os.path.exists(out_path)
    log_file = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()
        log_file.flush()

    print(f"Loaded {len(rows)} clue–answer pairs from {src_path}")
    print(f"Logging to {out_path}")
    print_rule()

    trial_counter = 0
    revealed_flag = 0

    def graceful_exit(signum, frame):
        print("\n\nCaught signal, saving and exiting...")
        log_file.flush()
        log_file.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, graceful_exit)

    while True:
        r = random.choice(rows)
        clue = r["clue"]
        answer_raw = r["answer"]
        answer_norm = normalize_answer(answer_raw)
        L = len(answer_norm)

        # Choose random mask
        filled_idxs = choose_mask_indices(L)
        shown_pattern = apply_mask(answer_raw, filled_idxs)

        # Build filled_pairs
        filled_pairs = [[i, answer_norm[i]] for i in filled_idxs]

        # Present
        print("\n" + "="*72)
        print(f"CLUE:   {clue}")
        print(f"PATTERN: {shown_pattern}   (length {L})")
        print("="*72)

        # Interaction loop for this clue
        revealed_flag = 0
        while True:
            user = input("Your answer (or :skip / :reveal / :quit): ").rstrip("\n")
            if user.strip() == ":quit":
                graceful_exit(None, None)
            elif user.strip() == ":reveal":
                print(f"ANSWER: {answer_raw}")
                revealed_flag = 1
                # Let the user then press Enter to proceed (records as revealed)
                user = ""
                break
            elif user.strip() == ":skip":
                user = ""
                break
            else:
                # Regular attempt
                break

        user_guess_raw = user
        user_guess_norm = normalize_answer(user_guess_raw)
        user_correct = int(user_guess_norm == answer_norm)

        num_remaining_correct = count_remaining_correct(user_guess_norm, answer_norm, filled_idxs)

        log_row = {
            "timestamp_utc": now_iso(),
            "session_id": session_id,
            "source_id": r["id"],
            "clue": clue,
            "answer_raw": answer_raw,
            "answer_norm": answer_norm,
            "answer_len": L,
            "clue_len": len(clue),
            "filled_indices": json.dumps(filled_idxs, ensure_ascii=False),
            "filled_pairs": json.dumps(filled_pairs, ensure_ascii=False),
            "shown_pattern": shown_pattern,
            "user_guess_raw": user_guess_raw,
            "user_guess_norm": user_guess_norm,
            "user_correct": user_correct,
            "revealed": int(revealed_flag),
            "num_remaining_correct": num_remaining_correct
        }

        writer.writerow(log_row)
        trial_counter += 1
        if trial_counter % SAVE_EVERY_N == 0:
            log_file.flush()

        # Lightweight progress indicator
        status = "✓ CORRECT" if user_correct else ("(revealed)" if revealed_flag else "✗ wrong/skip")
        print(f"Recorded: {status} | {trial_counter} total so far.")

if __name__ == "__main__":
    main()
""").strip()

# Save the generated script in the same directory as this file
script_dir = Path(__file__).parent
out_path = script_dir / "crossword_collector_app.py"
out_path.write_text(script, encoding="utf-8")

print(f"Generated crossword collector app at: {out_path}")
print(f"Run with: python {out_path}")
print("The app will automatically use crossword_data/crossword_train.csv if no path is provided.")
