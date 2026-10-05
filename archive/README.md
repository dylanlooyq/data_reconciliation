# Archive

The original first-study material, kept for reference and for git history. It is not
part of the current benchmark and is not maintained.

- `legacy_scripts/`: the numbered scripts (data generator, one script per method, batch-size optimiser, chart).
  They communicated through `results.pkl`; the current code replaces that with isolated runs and JSON results.
- `legacy_results/`: the pickled results from that study.
- `legacy_figures/`: the charts it produced.
- `reconciliation.ipynb`: the exploratory notebook the scripts grew out of.

Known problems in the original that the new harness fixes: PyArrow was averaged over 5 runs while
other methods ran once; the batch-size tuner picked the minimum of single noisy runs; no answer was
checked against ground truth; the generator's comment said 5 rows changed while the code changed 200.
