from pathlib import Path

RESULTS_PATH = Path("results.pkl")

if RESULTS_PATH.exists():
    RESULTS_PATH.unlink()  # deletes the file
    print("✅ 'results.pkl' has been deleted.")
else:
    print("⚠️ 'results.pkl' not found. Nothing to delete.")
