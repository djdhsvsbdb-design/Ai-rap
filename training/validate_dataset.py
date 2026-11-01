"""Basic JSONL dataset checks for RAP//AI."""
import json
from pathlib import Path

DATA_PATH = Path("training/data/rap_examples.jsonl")

def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")
    count = 0
    with DATA_PATH.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Line {line_no}: invalid JSON: {exc}")
            for field in ("instruction", "output"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise SystemExit(f"Line {line_no}: missing or empty {field}")
            count += 1
    if count < 2:
        raise SystemExit("Dataset is too small; add more original examples before training.")
    print(f"OK: {count} valid examples")

if __name__ == "__main__":
    main()
