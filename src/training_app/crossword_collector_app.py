#!/usr/bin/env python3
import csv
import json
import os
import random
import signal
import sys
import uuid
import select
import time
from datetime import datetime
from pathlib import Path

# ---------- Config ----------
SAVE_EVERY_N = 1   # append to file after every N trials
LOG_BASENAME = "crossword_session_LOG.csv"
VOIDED_IDS_FILE = "crossword_voided_ids.txt"
TIME_LIMIT_SECONDS = 15  # Time limit per ccranlue
RANDOM_SEED = None  # set an int for reproducibility, or None

# ---------- Helpers ----------
def get_timed_input(prompt, timeout_seconds):
    """Get user input with timeout. Returns (input_string, timed_out_flag)."""
    print(prompt, end='', flush=True)
    
    # Check if input is available within timeout
    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    
    if ready:
        # Input is available, read it
        user_input = sys.stdin.readline().rstrip('\n')
        return user_input, False
    else:
        # Timeout occurred
        print(f"\n⏰ TIME'S UP! ({timeout_seconds}s elapsed)")
        return "", True

def normalize_answer(ans: str) -> str:
    # Uppercase and strip spaces/hyphens; keep other characters to preserve abbreviations
    return ans.upper().replace(" ", "").replace("-", "").strip()

def choose_mask_indices(L: int) -> list:
    # Uniformly choose k in {0, 1, ..., L}; then choose k unique indices to reveal
    k = random.randint(0, L-1)
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

def load_voided_ids(project_root: Path) -> set:
    """Load the set of voided clue IDs from file."""
    voided_file_path = project_root / "data" / VOIDED_IDS_FILE
    try:
        with open(voided_file_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def save_voided_id(clue_id: str, project_root: Path):
    """Add a clue ID to the voided list."""
    voided_file_path = project_root / "data" / VOIDED_IDS_FILE
    with open(voided_file_path, "a", encoding="utf-8") as f:
        f.write(f"{clue_id}\n")
        f.flush()

def print_rule():
    print("-"*72)
    print(f"⏰ You have {TIME_LIMIT_SECONDS} seconds per clue!")
    print("Enter your answer and press <Enter>. Commands:")
    print("  :skip    -> skip this clue (records as incorrect, guess='')")
    print("  :reveal  -> show the answer, then press <Enter> to move on (records as revealed)")
    print("  :void    -> mark clue as void and remove from future sessions")
    print("  :quit    -> save & exit")
    print("-"*72)
    print("Usage: python crossword_collector.py [csv_path] [log_output.csv]")
    print("       If no csv_path provided, uses data/nytcrosswords.csv")
    print("       Filters for crosswords after 2015")
    print("-"*72)

# ---------- Main ----------
def main():
    # Get the project root directory (3 levels up from the generated script location)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent if script_dir.name == "training_app" else script_dir.parent.parent.parent
    default_csv_path = project_root / "data" / "nytcrosswords.csv"

    # Use command line argument if provided, otherwise use default path
    if len(sys.argv) >= 2:
        src_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) >= 3 else str(project_root / "data" / LOG_BASENAME)
    else:
        src_path = str(default_csv_path)
        out_path = str(project_root / "data" / LOG_BASENAME)
        print(f"No CSV path provided, using default: {src_path}")

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    if not os.path.exists(src_path):
        print(f"Error: source CSV not found: {src_path}")
        sys.exit(1)

    # Load dataset - updated for nytcrosswords.csv format
    rows = []
    
    # Try different encodings to handle various CSV formats
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        try:
            with open(src_path, "r", newline="", encoding=encoding) as f:
                reader = csv.DictReader(f)
                expected = {"Date", "Word", "Clue"}
                
                # Check if we have the expected columns
                if not expected.issubset(set(reader.fieldnames or [])):
                    print(f"Error: CSV must have columns {expected}, got {reader.fieldnames}")
                    sys.exit(1)
                
                print(f"Successfully opened CSV with {encoding} encoding")
                
                for r in reader:
                    # Check if required fields exist and are not empty
                    if not r.get("Date") or not r.get("Clue") or not r.get("Word"):
                        continue
                    
                    # Parse date and filter for years after 2015
                    try:
                        date_str = r["Date"]
                        # Try different date formats
                        if '/' in date_str:
                            # MM/DD/YYYY format
                            parts = date_str.split('/')
                            if len(parts) >= 3:
                                year = int(parts[2])
                            else:
                                continue
                        elif '-' in date_str:
                            # YYYY-MM-DD format
                            year = int(date_str.split('-')[0])
                        else:
                            # Try to parse as just a year
                            year = int(date_str[:4]) if len(date_str) >= 4 else 0
                        
                        if year <= 2015:
                            continue
                    except (ValueError, IndexError):
                        # Skip rows with invalid dates
                        continue
                    
                    word = r["Word"]
                    if not isinstance(word, str) or len(word.strip()) == 0:
                        continue
                    
                    # Create a unique ID from date and word
                    row_id = f"{r['Date']}_{word.replace(' ', '_')}"
                    rows.append({
                        "id": row_id, 
                        "clue": r["Clue"], 
                        "answer": word
                    })
                
                # If we got here, the encoding worked
                break
                
        except UnicodeDecodeError:
            print(f"Failed to read with {encoding} encoding, trying next...")
            continue
        except Exception as e:
            print(f"Error reading file with {encoding}: {e}")
            continue
    else:
        # If we tried all encodings and none worked
        print("Error: Could not read the CSV file with any supported encoding")
        sys.exit(1)

    if not rows:
        print("No usable rows found in the source CSV.")
        sys.exit(1)

    # Load voided IDs and filter them out
    voided_ids = load_voided_ids(project_root)
    original_count = len(rows)
    rows = [row for row in rows if row["id"] not in voided_ids]
    voided_count = original_count - len(rows)
    
    if voided_count > 0:
        print(f"Filtered out {voided_count} previously voided clues")
    
    if not rows:
        print("No clues remaining after filtering voided items.")
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
        "voided",                  # 0/1 (user marked as void)
        "timed_out",               # 0/1 (timer expired)
        "num_remaining_correct"    # among positions not pre-filled
    ]

    # If the file doesn't exist, create with header
    new_file = not os.path.exists(out_path)
    log_file = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()
        log_file.flush()

    print(f"Loaded {len(rows)} clue–answer pairs from {src_path} (post-2015 NYT crosswords)")
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
        print(f"⏰ TIME: {TIME_LIMIT_SECONDS} seconds")
        print("="*72)

        # Interaction loop for this clue with timer
        revealed_flag = 0
        voided_flag = 0
        timed_out_flag = 0
        
        try:
            user, timed_out_flag = get_timed_input("Your answer (or :skip / :reveal / :void / :quit): ", TIME_LIMIT_SECONDS)
            
            if timed_out_flag:
                # Timer expired
                print("⏰ Time expired!")
                user = ""
            elif user.strip() == ":quit":
                graceful_exit(None, None)
            elif user.strip() == ":reveal":
                print(f"ANSWER: {answer_raw}")
                revealed_flag = 1
                user = ""
            elif user.strip() == ":skip":
                user = ""
            elif user.strip() == ":void":
                print(f"Voiding clue: '{clue}' -> '{answer_raw}'")
                print("This clue will not appear in future sessions.")
                save_voided_id(r["id"], project_root)
                # Remove this item from the current session's rows list
                rows = [row for row in rows if row["id"] != r["id"]]
                voided_flag = 1
                user = ""
            # If it's a regular answer, we keep it as is
        except KeyboardInterrupt:
            graceful_exit(None, None)

        user_guess_raw = user
        user_guess_norm = normalize_answer(user_guess_raw)
        user_correct = int(user_guess_norm == answer_norm)
        if user_correct:
            print("Correct!")
        else:
            print(f"Incorrect! The correct answer was: {answer_raw}")

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
            "voided": int(voided_flag),
            "timed_out": int(timed_out_flag),
            "num_remaining_correct": num_remaining_correct
        }

        writer.writerow(log_row)
        trial_counter += 1
        if trial_counter % SAVE_EVERY_N == 0:
            log_file.flush()

        # Lightweight progress indicator
        if voided_flag:
            status = "🗑️ VOIDED"
        elif timed_out_flag:
            status = "⏰ TIMEOUT"
        elif user_correct:
            status = "✓ CORRECT"
        elif revealed_flag:
            status = "(revealed)"
        else:
            status = "✗ wrong/skip"
        print(f"Recorded: {status} | {trial_counter} total so far.")

if __name__ == "__main__":
    main()