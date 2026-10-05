# main.py
import subprocess

scripts = [
    # "01. wipe_results.py",
    # "02. data_generator.py",
    # "03. recon_pandas.py",
    # "04. recon_duck_db.py",
    # "05. recon_polars.py",
    # "06. recon_polars_streaming.py",
    "07. recon_polars_vectorized.py",
    # "08. recon_pyarrow.py",
    # "09. pyarrow_batch_optimizer.py",
    # "09. recon_optimised_pyarrow.py",
    "10. visualization_final_results.py",
]

for script in scripts:
    print(f"\n 🔥 Running {script}...")
    subprocess.run(["python", script], check=True)
