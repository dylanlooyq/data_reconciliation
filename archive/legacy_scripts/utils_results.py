# utils_results.py
import pickle, os
from pathlib import Path

RESULTS_PATH = Path("results.pkl")

def load_results(path: Path = RESULTS_PATH) -> dict:
    # Initialize if missing or empty
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError, AttributeError) as e:
        # File is corrupt or not a pickle — start fresh (and optionally back it up)
        backup = path.with_suffix(".corrupt.pkl")
        try:
            path.replace(backup)
        except Exception:
            pass
        print(f"[warn] results.pkl unreadable ({type(e).__name__}). "
              f"Backed up to {backup.name}. Starting with empty dict.")
        return {}

def save_results(results: dict, path: Path = RESULTS_PATH) -> None:
    # Atomic write to avoid leaving a 0-byte file if a crash occurs mid-write
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic on Windows & POSIX
